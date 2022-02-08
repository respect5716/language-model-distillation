import os
import hydra
import wandb
import deepspeed
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

from src.data import prepare_dataset
from src.distil import kl_div_loss, margin_loss
from src.model import get_param_groups, prepare_optimizer, prepare_scheduler


def prepare_model(config):
    teacher = AutoModelForMaskedLM.from_pretrained(**config.teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    cfg = AutoConfig.from_pretrained(**config.student)
    student = AutoModelForMaskedLM.from_config(cfg)

    teacher.eval()
    student.train()
    return teacher, student


class Lite(LightningLite):
    def run(self, config):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(config))
            wandb.init(project='language-model-distillation', config=OmegaConf.to_container(config))
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
        dataset = prepare_dataset(config, tokenizer, collator)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
        
        teacher, student = prepare_model(config)
        _ = teacher.to(self.device)
        if self.is_global_zero:
            wandb.watch(student, log='gradients', log_freq=10)

        params = get_param_groups(student, config.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, config.optimizer)
        scheduler = prepare_scheduler(optimizer, config.scheduler)
        _, optimizer = self.setup(student, optimizer)
        
        dataiter = iter(dataloader)
        pbar = tqdm(range(config.train.num_train_steps))
        for st in pbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = BatchEncoding(batch).to(self.device)

            tout = teacher(
                input_ids = batch.masked_input_ids,
                attention_mask = batch.attention_mask,
            )

            sout = student(
                input_ids = batch.masked_input_ids,
                attention_mask = batch.attention_mask,
                labels = batch.masked_labels
            )

            t = tout.logits.view(-1, teacher.config.vocab_size)
            s = sout.logits.view(-1, teacher.config.vocab_size)
            l = batch.input_ids.view(-1)

            val, _ = t.max(dim=-1)
            margined_val = val + config.train.margin
            t[torch.arange(len(t)), l] = margined_val

            hard_loss = sout.loss
            soft_loss = kl_div_loss(s, t, config.train.temperature)
            mse_loss = F.mse_loss(s, t)
            
            loss = 3 * soft_loss

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                wandb.log({'loss': loss, 'hard_loss': hard_loss, 'soft_loss': soft_loss, 'mse_loss': mse_loss})
                if (st + 1) % 10000 == 0:
                    student.base_model.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='margin_mlm')
def main(config: DictConfig):
    Lite(**config.lite).run(config)

if __name__ == '__main__':
    main()