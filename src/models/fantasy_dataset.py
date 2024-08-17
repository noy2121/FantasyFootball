import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import List, Dict

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler


class FantasyDataset:
    def __init__(self, data_dir: str, model_name: str, max_length: int = 512):
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Load train and test data
        self.train_data, self.test_data = self.load_data()

        # Initialize separate Dataset objects for train and test
        self.train_dataset = Dataset(pa.Table.from_pandas(self.train_data))
        self.test_dataset = Dataset(pa.Table.from_pandas(self.test_data))

        # Create a DatasetDict to hold both datasets
        self.dataset_dict = DatasetDict({
            'train': self.train_dataset,
            'test': self.test_dataset
        })

        # Optionally create a validation set from the train set
        # self.create_validation_set()

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data = self._load_csvs_from_dir(os.path.join(self.data_dir, 'train'))
        test_data = self._load_csvs_from_dir(os.path.join(self.data_dir, 'test'))
        return train_data, test_data

    @staticmethod
    def _load_csvs_from_dir(directory: str) -> pd.DataFrame:
        csv_files = ['clubs.csv', 'player.csv', 'events.csv', 'games.csv']
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(directory, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['source'] = file.split('.')[0]  # Add source column
                dataframes.append(df)
            else:
                print(f"Warning: {file} not found in {directory}")

        # Concatenate all DataFrames vertically
        combined = pd.concat(dataframes, axis=0, ignore_index=True)
        return combined

    def create_validation_set(self, valid_size: float = 0.1, seed: int = 42):
        train_valid = self.train_dataset.train_test_split(test_size=valid_size, seed=seed)
        self.train_dataset = train_valid['train']
        self.valid_dataset = train_valid['test']
        self.dataset_dict['valid'] = self.valid_dataset
        self.dataset_dict['train'] = self.train_dataset

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def get_tokenized_dataset(self, split: str) -> Dataset:
        return self.dataset_dict[split].map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=['text']  # Keep 'source' column for weighted sampling
        )

    def get_numpy_data(self, split: str, columns: List[str]) -> np.ndarray:
        return self.dataset_dict[split].to_pandas()[columns].to_numpy()

    def get_source_weights(self, split: str) -> Dict[str, float]:
        source_counts = self.dataset_dict[split].to_pandas()['source'].value_counts()
        total_samples = sum(source_counts)
        weights = {source: count / total_samples for source, count in source_counts.items()}
        return weights

    def __len__(self):
        return len(self.train_data)


class FantasyDataLoader:
    def __init__(self, dataset: FantasyDataset, split: str, batch_size: int = 32, num_workers: int = 4):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenized_dataset = self.dataset.get_tokenized_dataset(split)
        self.source_weights = self.dataset.get_source_weights(split)

        self.sampler = self._create_weighted_sampler()

    def _create_weighted_sampler(self) -> WeightedRandomSampler:
        sample_weights = [self.source_weights[source] for source in self.tokenized_dataset['source']]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tokenized_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
