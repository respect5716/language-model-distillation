import os
import torch
from datasets import load_dataset, concatenate_datasets


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
