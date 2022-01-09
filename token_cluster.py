import os
import hydra
import wandb
import deepspeed
import pandas as pd
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import BatchEncoding
from datasets import load_dataset, concatenate_datasets

from src.distil_utils import kl_div_loss
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
    cfg.num_labels = config.train.num_clusters
    student = AutoModelForTokenClassification.from_config(cfg)

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
        return teacher_outputs, student_outputs


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

        centroids = np.load(os.path.join(config.working_dir, 'centroids.npy'))
        centroids = torch.tensor(centroids).to(self.device)
        if self.is_global_zero:
            print(centroids.size())
        
        pbar = tqdm(range(config.train.num_train_steps))
        loader = iter(dataloader)
        for st in pbar:
            try:
                batch = next(loader)
            except:
                loader = iter(dataloader)
                batch = next(loader)
            batch = {k:v.to(self.device) for k,v in batch.items()}
            to, so = model(batch)

            th = to.last_hidden_state
            th = th.view(-1, th.size(-1))
            teacher_logits = torch.mm(th, centroids.T)
            student_logits = so.logits.view(-1, logits.size(-1))
            loss = kl_div_loss(student_logits, teacher_logits, config.train.temperature)

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                wandb.log({'loss': loss})
                if (st + 1) % 10000 == 0:
                    student.base_model.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='token_cluster')
def main(config: DictConfig):
    Lite(**config.lite).run(config)

if __name__ == '__main__':
    main()