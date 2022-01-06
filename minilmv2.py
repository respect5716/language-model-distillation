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
from transformers import BatchEncoding
from datasets import load_dataset, concatenate_datasets

from src.distil_utils import to_distill, minilm_loss, get_qkvs
from src.model_utils import get_param_groups, prepare_optimizer, prepare_scheduler


def transform(batch, tokenizer):
    input_ids = [[tokenizer.cls_token] + i.split() + [tokenizer.sep_token] for i in batch['text']]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in input_ids]
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def prepare_dataset(config, tokenizer):
    dataset = []
    for fname in config.data.files[config.data.lang]:
        _dataset = load_dataset('text', data_files=os.path.join(config.data_dir, f'{fname}.txt'))['train']
        dataset.append(_dataset)
    dataset = concatenate_datasets(dataset)
    dataset.set_transform(lambda batch: transform(batch, tokenizer))
    return dataset


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

class Model(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def forward(self, batch):
        teacher_outputs = self.teacher(**batch)
        student_outputs = self.student(**batch)
        return {'teacher': teacher_outputs, 'student': student_outputs}


class Lite(LightningLite):
    def run(self, config):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(config))
            wandb.init(project='language-model-distillation', config=config)
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        dataset = prepare_dataset(config, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)

        if self.is_global_zero:
            batch = next(iter(dataloader))
            for k, v in batch.items():
                print(k, v.size())


        teacher, student = prepare_model(config)
        model = Model(teacher, student)
        params = get_param_groups(student, config.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, config.optimizer)
        scheduler = prepare_scheduler(optimizer, config.scheduler)
        model, optimizer = self.setup(model, optimizer)
        
        pbar = tqdm(range(config.train.num_train_steps))
        loader = iter(dataloader)
        for st in pbar:
            try:
                batch = next(loader)
            except:
                loader = iter(dataloader)
                batch = next(loader)
            batch = {k:v.to(self.device) for k,v in batch.items()}
            _ = model(batch)

            teacher_qkv = get_qkvs(teacher)[config.train.teacher_layer_index] # (batch, head, seq, head_dim)
            student_qkv = get_qkvs(student)[config.train.student_layer_index] # (batch, head, seq, head_dim)

            loss_q = minilm_loss(teacher_qkv['q'], student_qkv['q'], config.train.num_relation_heads)
            loss_k = minilm_loss(teacher_qkv['k'], student_qkv['k'], config.train.num_relation_heads)
            loss_v = minilm_loss(teacher_qkv['v'], student_qkv['v'], config.train.num_relation_heads)
            loss = loss_q + loss_k + loss_v

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()


            if self.is_global_zero:
                wandb.log({'loss': loss, 'loss_q': loss_q, 'loss_k': loss_k, 'loss_v': loss_v})
                if (st + 1) % 10000 == 0:
                    student.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='minilmv2')
def main(config: DictConfig):
    Lite(**config.lite).run(config)

if __name__ == '__main__':
    main()