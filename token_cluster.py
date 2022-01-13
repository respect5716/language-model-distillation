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

from src.data import prepare_dataset
from src.model import get_param_groups, prepare_optimizer, prepare_scheduler
from src.distil import kl_div_loss, attention



class Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = AutoConfig.from_pretrained(**config.student)
        self.transformer = AutoModel.from_config(cfg)
        self.centroids = nn.Parameter(torch.rand(config.train.num_clusters, config.student.hidden_size))

    def forward(self, batch):
        out = self.transformer(**batch)
        out = out.last_hidden_state
        return out


class Teacher(nn.Module):
    def __init__(self, centroids, avg, config):
        super().__init__()
        self.centroids = centroids
        self.avg = avg
        self.config = config
        self.transformer = AutoModel.from_pretrained(**config.teacher)
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, batch):
        out = self.transformer(**batch)
        out = out.last_hidden_state
        out -= self.avg
        return out


class Lite(LightningLite):
    def run(self, config):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(config))
            wandb.init(project='language-model-distillation', config=OmegaConf.to_container(config))
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        dataset = prepare_dataset(config, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
        
        centroids = faiss.read_index(os.path.join(config.working_dir, 'kmeans'))
        centroids = [centroids.reconstruct(i) for i in range(config.train.num_clusters)]
        centroids = torch.tensor(np.stack(centroids, axis=0)).to(self.device)

        avg = np.load(os.path.join(config.working_dir, 'avg.npy'))
        avg = torch.tensor(avg).unsqueeze(0).to(self.device) # (1, 1, dim)

        teacher = Teacher(centroids, avg, config).to(self.device)
        student = Student(config)
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
            batch = {k:v.to(self.device) for k,v in batch.items()}

            out_t = teacher(batch) # (batch, seq, dim_t)
            out_s = student(batch) # (batch, seq, dim_s)

            out_t = out_t.view(-1, out_t.size(-1)) # (batch*seq, dim_t)
            out_s = out_s.view(-1, out_s.size(-1)) # (batch*seq, dim_s)
            
            score_t = torch.matmul(out_t, teacher.centroids.T) # (batch*seq, num_clusters)
            score_s = torch.matmul(out_s, student.centroids.T) # (batch*seq, num_clusters)
            global_loss = kl_div_loss(score_s, score_t, config.train.temperature)

            ranking = torch.argsort(-score_t, dim=-1) # (batch*seq, num_clusters)
            local_idxs = ranking[:, :config.train.local_k] # (batch*seq, local_k)

            local_out_t = out_t.unsqueeze(1) # (batch*seq, 1, dim)
            local_out_s = out_s.unsqueeze(1) # (batch*seq, 1, dim)

            local_centroids_t = teacher.centroids[local_idxs] # (batch*seq, local_k, dim_t)
            local_centroids_s = student.centroids[local_idxs] # (batch*seq, local_k, dim_s)


            attn_t = attention(local_out_t, local_centroids_t, config.train.num_local_heads) # (batch*seq, num_head, 1, local_k)
            attn_s = attention(local_out_s, local_centroids_s, config.train.num_local_heads) # (batch*seq, num_head, 1, local_k)
            local_loss = kl_div_loss(attn_s, attn_t, config.train.temperature)

            loss = global_loss * config.train.global_alpha + local_loss * config.train.local_alpha

            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                wandb.log({'loss': loss, 'global_loss': global_loss, 'local_loss': local_loss})
                if (st + 1) % 10000 == 0:
                    student.transformer.base_model.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='token_cluster')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()