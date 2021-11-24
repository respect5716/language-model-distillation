from datasets import load_dataset, concatenate_datasets

def prepare_text_dataset(dataset_names, data_dir='../data'):
    dataset = []
    for dname in dataset_names:
        _dataset = load_dataset('text', data_files=os.path.join(data_dir, f'{dname}.txt'))['train']
        dataset.append(_dataset)
    
    return concatenate_datasets(dataset)

def transform(batch):
    chunks = []
    for text in batch['text']:
        if len(text) > 2048:
            idx = random.randint(0, len(text) - 1024)
            text = text[idx:idx+1024].strip()
        chunks.append(text)
    
    return tokenizer(chunks, padding='max_length', truncation=True, max_length=512, return_tensors='pt')