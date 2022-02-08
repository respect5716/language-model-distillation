import os
import torch
from datasets import load_dataset, concatenate_datasets


def transform(batch, tokenizer, collator):
    input_ids = [[tokenizer.cls_token] + i.split() + [tokenizer.sep_token] for i in batch['text']]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in input_ids]
    input_ids = torch.tensor(input_ids)
    inputs = {}
    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = torch.ones_like(input_ids)
    
    if collator is not None:
        mlm = collator(input_ids.tolist())
        inputs['masked_input_ids'] = mlm['input_ids']
        inputs['masked_labels'] = mlm['labels']
    
    return inputs


def prepare_dataset(config, tokenizer, collator=None):
    dataset = []
    for fname in config.data.files[config.data.lang]:
        _dataset = load_dataset('text', data_files=os.path.join(config.data_dir, f'{fname}.txt'))['train']
        dataset.append(_dataset)
    dataset = concatenate_datasets(dataset)
    dataset.set_transform(lambda batch: transform(batch, tokenizer, collator))
    return dataset