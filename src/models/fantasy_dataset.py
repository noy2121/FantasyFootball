import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List
from functools import partial, reduce

import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer
import numpy as np


class WeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class MultiDataset(Dataset):
    def __init__(self, data_dir: str, filenames: str, tokenizer, max_length=512):

        self.dataframes = self.load_datasets(data_dir, filenames)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_keys = list(datasets.keys())

        # Calculate total length and weights
        self.dataset_lengths = [len(dataset) for dataset in datasets.values()]
        self.total_length = sum(self.dataset_lengths)
        self.weights = [length / self.total_length for length in self.dataset_lengths]

        # Create an index mapping
        self.index_map = []
        for key in self.dataset_keys:
            self.index_map.extend([(key, i) for i in range(len(self.datasets[key]))])

    @staticmethod
    def load_datasets(data_dir: str, filenames: List[str] = None) -> Dict[str, pd.DataFrame]:
        if filenames == 'all':  # load all datasets
            dfs = {str(Path(fn).stem): pd.read_csv(f'{data_dir}/{fn}') for fn in
                   os.listdir(data_dir) if Path(fn).suffix == '.csv'}
        else:
            dfs = {str(Path(fn).stem): pd.read_csv(f'{data_dir}/{fn}') for fn in filenames
                   if Path(f'{data_dir}/{fn}').exists() and Path(fn).suffix == '.csv'}
        print(f'Load {len(dfs)} different datasets:')
        for k, v in dfs.items():
            print(f'\t- {k}: shape {v.shape}')

        return dfs

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_key, item_idx = self.index_map[idx]
        text = self.datasets[dataset_key][item_idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'dataset_key': dataset_key
        }


class MultiDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        sampler = WeightedRandomSampler(
            weights=self.dataset.weights,
            num_samples=len(self.dataset),
            replacement=True
        )

        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers
        )

        for batch in data_loader:
            yield batch


# Usage example:
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

datasets = {
    'players': ["Player 1 info", "Player 2 info", ...] * 10000,  # 10k samples
    'clubs': ["Club 1 info", "Club 2 info", ...] * 100,  # 100 samples
    'games': ["Game 1 info", "Game 2 info", ...] * 1000,  # 1k samples
    'events': ["Event 1 info", "Event 2 info", ...] * 5000  # 5k samples
}

multi_dataset = MultiDataset(datasets, tokenizer)
data_loader = MultiDataLoader(multi_dataset, batch_size=32)

# Training loop
for batch in data_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    dataset_keys = batch['dataset_key']
    # ... process batch ...

# Print sampling statistics
sample_count = {key: 0 for key in datasets.keys()}
total_samples = 10000  # number of samples to check

for _ in range(total_samples):
    idx = next(iter(WeightedRandomSampler(multi_dataset.weights, 1)))
    sample_count[multi_dataset.index_map[idx][0]] += 1

print("Sampling statistics:")
for key, count in sample_count.items():
    print(f"{key}: {count / total_samples:.2%}")


class FantasyDataset:
    def __init__(self, data_dir: str, filenames: List[str] = None, model_name: str = 'bert-base-uncased'):
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dfs_dict = self.load_datasets(data_dir, filenames)  # Dict[str, pd.DataFrame]
        self.combined_data = self._combine_data()
        self.dataset = self._create_dataset()



    def _combine_data(self) -> pd.DataFrame:

        dfs_to_merge = [df for df in self.dfs_dict.values()]
        combined = reduce(lambda left, right: pd.merge(left, right, ))
        return combined

    def _create_dataset(self) -> DatasetDict:
        table = pa.Table.from_pandas(self.combined_data)
        dataset = Dataset(table)

        # Use a more efficient splitting method
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        train_valid = train_test['train'].train_test_split(test_size=0.1, seed=42)

        return DatasetDict({
            'train': train_valid['train'],
            'validation': train_valid['test'],
            'test': train_test['test']
        })

    def tokenize_function(self, examples: Dict[str, List], text_fields: List[str]) -> Dict[str, torch.Tensor]:
        texts = [' '.join([str(examples[field][i]) for field in text_fields])
                 for i in range(len(examples[text_fields[0]]))]

        # Batch tokenization for efficiency
        tokenized = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")

        return {k: v.squeeze(0) for k, v in tokenized.items()}

    def get_dataset(self, tokenize: bool = True, text_fields: List[str] = None) -> DatasetDict:
        if tokenize:
            if text_fields is None:
                text_fields = ['player_name', 'club_name', 'event_type']  # Example fields

            # Use partial function for efficiency
            tokenize_fn = partial(self.tokenize_function, text_fields=text_fields)

            return self.dataset.map(
                tokenize_fn,
                batched=True,
                batch_size=1000,  # Adjust based on your memory constraints
                num_proc=4,  # Adjust based on your CPU
                remove_columns=self.dataset["train"].column_names
            )
        return self.dataset

    def get_dataloader(self, split: str, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        dataset = self.get_dataset(tokenize=True)[split]
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,  # Adjust based on your CPU
            pin_memory=True  # Helps with GPU memory transfer
        )

    def get_numpy_data(self, split: str, columns: List[str]) -> np.ndarray:
        return self.dataset[split].to_pandas()[columns].to_numpy()


# Usage example
fantasy_dataset = FantasyDataset('path/to/your/data/directory')

# Get tokenized datasets
tokenized_datasets = fantasy_dataset.get_dataset(tokenize=True)

# Get a PyTorch DataLoader
train_dataloader = fantasy_dataset.get_dataloader('train')

# Get numpy array for specific columns
train_numpy = fantasy_dataset.get_numpy_data('train', ['player_id', 'club_id', 'game_id'])

# Use with a transformers model
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()