import os
import numpy as np

import torch
import pytorch_lightning as pl

from transformers import BatchEncoding
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.teacher_name_or_model_path)
        
    def setup(self, stage=None):
        dataset = []
        for dname in self.hparams.datasets:
            _dataset = load_dataset('text', data_files=os.path.join(self.hparams.data_dir, f'{dname}.txt'))['train']
            dataset.append(_dataset)

        self.dataset = concatenate_datasets(dataset)
        self.dataset.set_transform(lambda batch: transform(batch, self.tokenizer, self.hparams.max_seq_length))
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    
    
def transform(batch, tokenizer, max_length):
    new_batch = []
    for text in batch['text']:
        text = slice_text(text)
        new_batch.append(text)
    
    return tokenizer(new_batch, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')


def slice_text(text, max_char_length=1024):
    if len(text) > max_char_length:
        idx = np.random.randint(low=0, high=len(text)-max_char_length)
        text = text[idx : idx+max_char_length]
    return text

def collate_fn(batch):
    new_batch = {}
    keys = batch[0].keys()
    for k in keys:
        new_batch[k] = torch.stack([b[k] for b in batch], dim=0)
    new_batch = BatchEncoding(new_batch)
    return new_batch