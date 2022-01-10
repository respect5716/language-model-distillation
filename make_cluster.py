import os
import faiss
import hydra
import wandb
import shutil
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



@hydra.main(config_path='conf', config_name='token_cluster')
def main(config: DictConfig):
    centroids_dir = os.path.join(config.working_dir, 'centroids')
    if os.path.isdir(centroids_dir):
        shutil.rmtree(centroids_dir)
    os.makedirs(centroids_dir)

    db_path = os.path.join(config.working_dir, config.train.db_path)
    if os.path.isfile(db_path):
        db = np.load(db_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        dataset = prepare_dataset(config, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        batch = next(iter(dataloader))

        model = AutoModel.from_pretrained(**config.teacher)
        _ = model.cuda().eval()
        for param in model.parameters():
            param.requires_grad = False

        db = []
        for st, batch in tqdm(enumerate(dataloader)):
            batch = {k:v.to(model.device) for k,v in batch.items()}
            out = model(**batch)
            hs = out.last_hidden_state.cpu()
            for i in range(hs.size(0)):
                idx = torch.randint(0, hs.size(1), (5,))
                db.append(hs[i][idx])
            
            if len(db) >= config.train.db_size:
                break
        
            db = torch.cat(db, dim=0)
            db = db.numpy()
            np.save('db.npy', db)
    
    print('DB size: ', db.shape)
    
    cluster_dim = db.shape[1] // config.train.num_sections
    for i in range(config.train.num_sections):
        kmeans = faiss.Kmeans(cluster_dim, config.train.num_clusters, niter=30, nredo=5, spherical=True, verbose=True, gpu=True)
        X = db[:, i*cluster_dim:(i+1)*cluster_dim]
        kmeans.train(np.ascontiguousarray(X))
        np.save(os.path.join(centroids_dir, f'centroids_{i:02d}.npy'), kmeans.centroids)


if __name__ == '__main__':
    main()