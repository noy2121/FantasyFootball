import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from models.rag_dataset import SeasonSpecificRAG


class FantasyDataset:
    def __init__(self, data_dir: str, model_name: str, rag_dir: str, max_length: int = 512):
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Load train and test data
        self.train_data, self.test_data = self.load_data()

        # Create a DatasetDict to hold both datasets
        self.dataset_dict = DatasetDict({
            'train': self.tokenize_and_prepare_dataset(self.train_data),
            'test': self.tokenize_and_prepare_dataset(self.test_data)
        })

        # initialize RAG
        self.rag = SeasonSpecificRAG.load(rag_dir)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = self._load_json_file('train.json')
        test_data = self._load_json_file('test.json')
        return train_data, test_data

    def _load_json_file(self, filename: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, filename)
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                sample_id, sample = json.loads(line).popitem()
                parsed_sample = self._parse_sample(sample)
                parsed_sample['sample_id'] = sample_id
                data.append(parsed_sample)
            return pd.DataFrame(data)

    def _parse_sample(self, sample: str) -> Dict[str, List[str]]:
        lines = sample.strip().split('\n')
        matches = lines[0].split('matches: ')[1].strip('[]').split(', ')
        teams = lines[0].split('matches: ')[1].strip('[]').strip(',').split(' vs ')
        round_info = lines[1].split('round: ')[1]
        season = lines[2].split('season: ')[1]
        date = lines[3].split('date: ')[1]
        return {
            'matches': matches,
            'teams': teams,
            'round': round_info,
            'season': season,
            'date': date,
            'text': sample  # Keep the original text for tokenization
        }

    def tokenize_and_prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        tokenized = self.tokenizer(
            df['text'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'matches': df['matches'].tolist(),
            'teams': df['teams'].tolist(),
            'round': df['round'].tolist(),
            'season': df['season'].tolist(),
            'date': df['date'].tolist(),
            'text': df['text'].tolist()  # Keep the original text
        })

        return dataset

    def create_validation_set(self, valid_size: float = 0.1, seed: int = 42):
        train_valid = self.train_dataset.train_test_split(test_size=valid_size, seed=seed)
        self.train_dataset = train_valid['train']
        self.valid_dataset = train_valid['test']
        self.dataset_dict['valid'] = self.valid_dataset
        self.dataset_dict['train'] = self.train_dataset

    def retrieve_relevant_info(self, query: str, date: str, season: str, k: int = 5) -> List[str]:
        return self.rag.retrieve_relevant_info(query, date, season, k)

    def __len__(self):
        return len(self.train_data)
