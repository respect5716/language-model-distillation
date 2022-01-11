import os
import hydra
import wandb
import faiss
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
from src.data_utils import prepare_dataset

class Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = AutoConfig.from_pretrained(**config.student)
        cfg.num_labels = config.train.num_clusters
        self.transformer = AutoModelForTokenClassification.from_config(cfg)

    def forward(self, batch):
        out = self.transformer(**batch)
        return out.logits


class Teacher(nn.Module):
    def __init__(self, kmeans, avg, config):
        super().__init__()
        self.kmeans = kmeans
        self.avg = avg
        self.config = config
        self.transformer = AutoModel.from_pretrained(**config.teacher)
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, batch):
        out = self.transformer(**batch)
        out = out.last_hidden_state
        out = out.view(-1, out.size(-1))
        out -= self.avg

        D, I = self.kmeans.search(out.cpu().numpy(), self.config.train.num_clusters)
        logits = torch.tensor(D).to(batch['input_ids'].device)
        return logits if self.config.train.spherical else -logits


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
            wandb.init(project='language-model-distillation', config=OmegaConf.to_container(config))
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        dataset = prepare_dataset(config, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
        
        kmeans = faiss.read_index(os.path.join(config.working_dir, 'kmeans'))
        avg = np.load(os.path.join(config.working_dir, 'avg.npy'))
        avg = torch.tensor(avg).to(self.device)

        teacher = Teacher(kmeans, avg, config)
        student = Student(config)
        model = Model(teacher, student)
        wandb.watch(student, log='gradients', log_freq=10)

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
            teacher_logits, student_logits = model(batch)
            loss = kl_div_loss(student_logits, teacher_logits, config.train.temperature)
            
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