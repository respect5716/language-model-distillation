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

from src.distil_utils import to_distill, minilm_loss, get_qkvs
from src.model_utils import get_param_groups, prepare_optimizer, prepare_scheduler
from src.data_utils import prepare_dataset



@hydra.main(config_path='conf', config_name='token_cluster')
def main(config: DictConfig):
    db_path = os.path.join(config.working_dir, config.train.db_path)
    if not os.path.isfile(db_path):
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

    db = np.load(db_path)
    avg = db.mean(axis=0, keepdims=True)
    np.save(os.path.join(config.working_dir, 'avg.npy'), avg)
    db -= avg
    
    kmeans = faiss.Kmeans(db.shape[1], config.train.num_clusters, niter=config.train.niter, nredo=config.train.nredo, spherical=config.train.spherical, verbose=True, gpu=True)
    kmeans.train(db)
    faiss.write_index(kmeans.index, os.path.join(config.working_dir, 'kmeans'))


if __name__ == '__main__':
    main()