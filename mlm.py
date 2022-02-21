import os
import hydra
import wandb
import deepspeed
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
from src.distil import kl_div_loss
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
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
        dataset = prepare_dataset(config, tokenizer, collator)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)

        if self.is_global_zero:
            batch = next(iter(dataloader))
            for k, v in batch.items():
                print(k, v.size())

        teacher, student = prepare_model(config)
        if self.is_global_zero and not config.debug:
            wandb.init(project='language-model-distillation', config=OmegaConf.to_container(config))
            wandb.watch(student, log='gradients', log_freq=10)

        params = get_param_groups(student, config.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, config.optimizer)
        scheduler = prepare_scheduler(optimizer, config.scheduler)
        _, optimizer = self.setup(student, optimizer)
        _ = teacher.to(self.device)

        dataiter = iter(dataloader)
        pbar = tqdm(range(config.train.num_train_steps))
        for st in pbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = BatchEncoding(batch).to(self.device)

            to = teacher(input_ids=batch.masked_input_ids, attention_mask=batch.attention_mask)
            so = student(input_ids=batch.masked_input_ids, attention_mask=batch.attention_mask, labels=batch.masked_labels)

            loss, log = 0., {}
            if config.train.alpha_mlm > 0:
                mlm_loss = so.loss
                loss += config.train.alpha_mlm * mlm_loss
                log['mlm_loss'] = mlm_loss.item()

            if config.train.alpha_distil > 0:
                distil_loss = kl_div_loss(so.logits, to.logits, config.train.temperature)
                loss += config.train.alpha_distil * distil_loss
                log['distil_loss'] = distil_loss.item()

            log['loss'] = loss.item()

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if config.debug:
                print(log)
                break

            if self.is_global_zero:
                wandb.log(log)
                if (st + 1) % 10000 == 0:
                    student.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='mlm')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()