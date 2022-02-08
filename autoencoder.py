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

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

from src.data import prepare_dataset
from src.distil import kl_div_loss, margin_loss, to_distill, get_feat
from src.model import get_param_groups, prepare_optimizer, prepare_scheduler
from train_autoencoder import AutoEncoder, MultiAutoEncoder

def prepare_model(config):
    teacher = AutoModelForMaskedLM.from_pretrained(**config.teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    cfg = AutoConfig.from_pretrained(**config.student)
    student = AutoModel.from_config(cfg)

    teacher = to_distill(teacher)
    student = to_distill(student)

    teacher.eval()
    student.train()
    return teacher, student

def contrastive_loss(h1, h2):
    mm = torch.matmul(h1, h2.transpose(-1, -2)) # (4096 * 4096)
    labels = torch.arange(h1.size(0)).to(h1.device)
    loss = F.cross_entropy(mm, labels)
    return loss

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
        lm_head = teacher.lm_head
        if self.is_global_zero:
            wandb.watch(student, log='gradients', log_freq=10)

        params = get_param_groups(student, config.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, config.optimizer)
        scheduler = prepare_scheduler(optimizer, config.scheduler)
        _, optimizer = self.setup(student, optimizer)

        ae = MultiAutoEncoder(teacher.config.hidden_size, teacher.config.intermediate_size, config.student.hidden_size, config.student.intermediate_size)
        model_name = config.model_name_or_path.replace('/', '-')
        ae.load_state_dict(torch.load(os.path.join(config.working_dir, f'{model_name}_autoencoder.pt')))
        for param in ae.parameters():
            param.requires_grad = False
        _ = ae.to(self.device)

        dataiter = iter(dataloader)
        pbar = tqdm(range(config.train.num_train_steps))
        for st in pbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = BatchEncoding(batch).to(self.device)

            tout = teacher.base_model(
                input_ids = batch.input_ids,
                attention_mask = batch.attention_mask,
            )

            sout = student.base_model(
                input_ids = batch.input_ids,
                attention_mask = batch.attention_mask,
            )

            tfeat = get_feat(teacher, -1)
            tfeat.append(tout.last_hidden_state)
            sfeat = get_feat(student, -1)
            sfeat.append(sout.last_hidden_state)

            enc, dec = [], []
            for i in range(len(tfeat)):
                enc.append(ae.aes[i].encoder(tfeat[i]))
                dec.append(ae.aes[i].decoder(sfeat[i]))

            mse_loss, cos_loss = 0., 0.
            for i in range(len(tfeat)):
                t, s, e, d = tfeat[i], sfeat[i], enc[i], dec[i]
                t = t.view(-1, t.size(-1))
                s = s.view(-1, s.size(-1))
                e = e.view(-1, e.size(-1))
                d = d.view(-1, d.size(-1))

                mse_loss += F.mse_loss(s, e)
                mse_loss += F.mse_loss(d, t)
                
                target = torch.ones(t.size(0)).to(self.device)
                cos_loss += F.cosine_embedding_loss(s, e, target)
                cos_loss += F.cosine_embedding_loss(d, t, target)
    
            loss = mse_loss + cos_loss
            
            optimizer.zero_grad()
            self.backward(loss)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                wandb.log({'loss': loss, 'mse_loss': mse_loss, 'cos_loss': cos_loss})
                if (st + 1) % 10000 == 0:
                    student.base_model.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))
                    tokenizer.save_pretrained(os.path.join(config.save_dir, f'{st+1:06d}'))


@hydra.main(config_path='conf', config_name='autoencoder')
def main(config: DictConfig):
    Lite(**config.lite).run(config)

if __name__ == '__main__':
    main()