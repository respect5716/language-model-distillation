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
from src.distil import scaled_dot_product, kl_div_loss, attention

class AutoEncoder(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(student_dim, teacher_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    

class MultiAutoEncoder(nn.Module):
    def __init__(self, teacher_hidden, teacher_intermediate, student_hidden, student_intermediate):
        super().__init__()
        self.aes = nn.ModuleList([
            AutoEncoder(teacher_hidden, student_hidden),
            AutoEncoder(teacher_hidden, student_hidden),
            AutoEncoder(teacher_hidden, student_hidden),
            AutoEncoder(teacher_hidden, student_hidden),
            AutoEncoder(teacher_intermediate, student_intermediate),
            AutoEncoder(teacher_hidden, student_hidden),
            AutoEncoder(teacher_hidden, student_hidden),
        ])
    
    def forward(self, t):
        res = []
        for i in range(len(t)):
            _res = self.aes[i](t[i])
            res.append(_res)
        return res

@hydra.main(config_path='conf', config_name='autoencoder')
def main(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    dataset = prepare_dataset(config, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
    dataiter = iter(dataloader)

    teacher = AutoModel.from_pretrained(**config.teacher)
    for param in teacher.parameters():
        param.requires_grad = False
    _ = teacher.to('cuda')

    ae = AutoEncoder(teacher.config.hidden_size, config.student.hidden_size).to('cuda')
    optim = torch.optim.Adam(ae.parameters(), lr=5e-4)

    pbar = tqdm(range(config.train.num_ae_steps))
    for st in pbar:
        batch = next(dataiter)
        batch = BatchEncoding(batch)
        batch = batch.to(teacher.device)
        
        tout = teacher(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        th = tout.last_hidden_state.view(-1, teacher.config.hidden_size)
        out = ae(th)
        
        if config.train.ae_loss == 'l2':
            loss = F.mse_loss(out, th)
        elif config.train.ae_loss == 'l1':
            loss = F.l1_loss(out, th)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if st % 100 == 0:
            print(f'st {st:06d} | loss {loss:.3f}')

    model_name = config.model_name_or_path.replace('/', '-')
    torch.save(ae.state_dict(), os.path.join(config.working_dir, f'{model_name}_autoencoder.pt'))


if __name__ == '__main__':
    main()