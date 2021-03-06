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

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

from src.data import prepare_dataset
from src.distil import to_distill, attention, kl_div_loss
from src.model import get_param_groups, prepare_optimizer, prepare_scheduler


def prepare_model(config):
    teacher = AutoModel.from_pretrained(**config.teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    cfg = AutoConfig.from_pretrained(**config.student)
    student = AutoModel.from_config(cfg)

    teacher = to_distill(teacher)
    student = to_distill(student)
    teacher.eval()
    student.train()
    return teacher, student


def get_qkvs(model):
    attns = [l.attention.self for l in model.base_model.encoder.layer]
    qkvs = [{'q': a.q, 'k': a.k, 'v': a.v} for a in attns]    
    return qkvs

def minilm_loss(s, t, num_heads, attention_mask=None, temperature=1.0):
    attn_t = attention(t, t, num_heads, attention_mask)
    attn_s = attention(s, s, num_heads, attention_mask)
    loss = kl_div_loss(attn_s, attn_t, temperature=temperature)
    return loss


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

            to = teacher.base_model(input_ids = batch.input_ids, attention_mask = batch.attention_mask)
            so = student.base_model(input_ids = batch.input_ids, attention_mask = batch.attention_mask)

            teacher_qkv = get_qkvs(teacher)[config.train.teacher_layer_index] # (batch, head, seq, head_dim)
            student_qkv = get_qkvs(student)[config.train.student_layer_index] # (batch, head, seq, head_dim)

            loss_q = minilm_loss(student_qkv['q'], teacher_qkv['q'], config.train.num_relation_heads)
            loss_k = minilm_loss(student_qkv['k'], teacher_qkv['k'], config.train.num_relation_heads)
            loss_v = minilm_loss(student_qkv['v'], teacher_qkv['v'], config.train.num_relation_heads)
            loss = loss_q + loss_k + loss_v

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            log = {'loss': loss.item(), 'loss_q': loss_q.item(), 'loss_k': loss_k.item(), 'loss_v': loss_v.item()}
            if config.debug:
                print(log)
                break

            if self.is_global_zero:
                wandb.log(log)
                if (st + 1) % 10000 == 0:
                    student.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='minilmv2')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()