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


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_prob):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.linear(self.dropout(x))


class Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = AutoConfig.from_pretrained(**config.student)
        cfg.num_labels = config.train.num_clusters
        self.transformer = AutoModel.from_config(cfg)
        self.classifiers = nn.ModuleList([
            Classifier(config.student.hidden_size, config.train.num_clusters, config.student.hidden_dropout_prob)
            for _ in range(config.train.num_sections)    
        ])

    def forward(self, batch):
        out = self.transformer(**batch)
        out = out.last_hidden_state
        out = [cl(out) for cl in self.classifiers]
        out = [o.view(-1, self.config.train.num_clusters) for o in out]
        return out


class Teacher(nn.Module):
    def __init__(self, centroids, config):
        super().__init__()
        self.config = config
        self.centroids = centroids
        self.transformer = AutoModel.from_pretrained(**config.teacher)
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.cluster_dim = self.transformer.config.hidden_size // config.train.num_sections

    def forward(self, batch):
        out = self.transformer(**batch)
        out = out.last_hidden_state
        out = out.view(-1, out.size(-1))

        logits = []
        for idx, cent in enumerate(self.centroids):
            X = out[:, idx*self.cluster_dim:(idx+1)*self.cluster_dim]
            logits.append(torch.mm(X, cent.T))
        return logits


class Model(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def forward(self, batch):
        teacher_outputs = self.teacher(batch)
        student_outputs = self.student(batch)
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

        centroids = [np.load(os.path.join(config.working_dir, 'centroids', f'centroids_{i:02d}.npy')) for i in range(config.train.num_sections)]
        centroids = [torch.tensor(cent).to(self.device) for cent in centroids]

        teacher = Teacher(centroids, config)
        student = Student(config)
        model = Model(teacher, student)
        params = get_param_groups(student, config.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, config.optimizer)
        scheduler = prepare_scheduler(optimizer, config.scheduler)
        model, optimizer = self.setup(model, optimizer)

        if self.is_global_zero:
            print(centroids[0].size())
        
        pbar = tqdm(range(config.train.num_train_steps))
        loader = iter(dataloader)
        for st in pbar:
            try:
                batch = next(loader)
            except:
                loader = iter(dataloader)
                batch = next(loader)
            batch = {k:v.to(self.device) for k,v in batch.items()}
            teacher_logits, student_logits = model(batch)

            loss = 0.
            for tl, sl in zip(teacher_logits, student_logits):
                loss += kl_div_loss(sl, tl, config.train.temperature) * config.train.alpha

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                wandb.log({'loss': loss})
                if (st + 1) % 10000 == 0:
                    student.transformer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='token_cluster')
def main(config: DictConfig):
    Lite(**config.lite).run(config)

if __name__ == '__main__':
    main()