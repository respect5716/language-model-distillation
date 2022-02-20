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
from src.distil import kl_div_loss, to_distill
from src.model import get_param_groups, prepare_optimizer, prepare_scheduler

class Student(nn.Module):
    def __init__(self, transformer, teacher_hidden_size):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.base_model = self.transformer.base_model
        self.upsampler = nn.ModuleList([nn.Linear(self.config.hidden_size, teacher_hidden_size) for _ in range(self.config.num_hidden_layers+1)])

    def forward(self, input_ids, attention_mask):
        return self.transformer(input_ids, attention_mask)


def prepare_model(config):
    teacher = AutoModelForMaskedLM.from_pretrained(**config.teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    cfg = AutoConfig.from_pretrained(**config.student)
    _student = AutoModelForMaskedLM.from_config(cfg)
    student = Student(_student, teacher.config.hidden_size)

    teacher = to_distill(teacher)
    student = to_distill(student)

    teacher.eval()
    student.train()
    return teacher, student


def get_layer_mapper(student_num_layers, teacher_num_layers):
    share = teacher_num_layers // student_num_layers
    layer_mapper = {0:0}
    for s in range(1, student_num_layers+1):
        layer_mapper[s] = s * share
    return layer_mapper

def get_attns(model):
    attns = [l.attention.self.attn for l in model.base_model.encoder.layer]
    return attns


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

        layer_mapper = get_layer_mapper(student.config.num_hidden_layers, teacher.config.num_hidden_layers)
        if self.is_global_zero:
            print(layer_mapper)
        
        dataiter = iter(dataloader)
        pbar = tqdm(range(config.train.num_train_steps))
        for st in pbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = BatchEncoding(batch).to(self.device)

            to = teacher(input_ids = batch.input_ids, attention_mask = batch.attention_mask)
            so = student(input_ids = batch.input_ids, attention_mask = batch.attention_mask)
            tattns = get_attns(teacher)
            sattns = get_attns(student)

            loss = 0.
            log = {}

            if config.train.alpha_hidden > 0:
                hidden_loss = 0.
                for si, ti in layer_mapper.items():
                    th, sh = to.hidden_states[ti], so.hidden_states[si]
                    sh = student.upsampler[si](sh)
                    hidden_loss += F.mse_loss(sh, th)
                loss += config.train.alpha_hidden * hidden_loss
                log['hidden_loss'] = hidden_loss.item()

            if config.train.alpha_attn > 0:
                attn_loss = 0.
                for ta, sa in zip(tattns, sattns):
                    attn_loss += F.mse_loss(sa, ta) / (ta.size(0) * ta.size(1))
                loss += config.train.alpha_attn * attn_loss
                log['attn_loss'] = attn_loss.item()

            if config.train.alpha_pred > 0:
                pred_loss = kl_div_loss(so.logits, to.logits, config.train.temperature)
                loss += config.train.alpha_pred * pred_loss
                log['pred_loss'] = pred_loss.item()

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


@hydra.main(config_path='conf', config_name='tinybert')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()