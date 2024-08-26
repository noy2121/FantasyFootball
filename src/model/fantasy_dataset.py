import os
import json
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple

from datasets import Dataset, DatasetDict


class FantasyDataset:
    def __init__(self, data_dir: str, max_length: int = 512):
        self.data_dir = data_dir
        self.max_length = max_length

        # Load train and test data
        self.train_data, self.test_data = self.load_data()

        # Create a DatasetDict to hold both datasets
        self.dataset_dict = DatasetDict({
            'train': self.prepare_dataset(self.train_data),
            'test': self.prepare_dataset(self.test_data)
        })

        self.matches_per_round = {
            "final": 1,
            "semi-final": 2,
            "quarter-final": 4,
            "round of 16": 8,
            "group stage": 16
        }

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

        matches = re.findall(r'([\w\s]+) vs ([\w\s]+)', sample)
        kn_round = re.search(r'round: ([\w\s-]+)', sample)
        season = re.search(r'season: (\d{4}-\d{2}-\d{2})', sample)
        date_str = re.search(r'date: (\d{4}-\d{2}-\d{2})', sample)
        teams = [team for match in matches for team in match]

        valid, msg = self.validate_sample(matches, kn_round, date_str)
        if not valid:
            raise ValueError(f'{msg}')

        return {
            'matches': matches,
            'teams': teams,
            'round': kn_round,
            'season': season,
            'date': date_str,
            'text': sample  # Keep the original text for tokenization
        }

    def validate_sample(self, matches, kn_round, date_str):
        knockout_round = kn_round.lower().strip()

        if knockout_round not in self.matches_per_round:
            return False, f"Invalid round: {knockout_round}. Expected one of {', '.join(self.matches_per_round.keys())}."

        if len(matches) != self.matches_per_round[knockout_round]:
            return False, f"Number of matches ({len(matches)}) does not match the expected number for {knockout_round} ({self.matches_per_round[knockout_round]})."

        # Validate number of teams
        teams = set(team for match in matches for team in match.split(' vs '))
        if len(teams) != 2 * len(matches):
            return False, f"Number of unique teams ({len(teams)}) should be exactly twice the number of matches ({len(matches)})."

        # Validate date
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            start_date = datetime(2018, 8, 10)
            end_date = datetime.now()
            if not (start_date <= date <= end_date):
                return False, f"Date {date_str} is not within the valid range (10/08/2018 to today)."
        except ValueError:
            return False, f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD."

        return True, "Input is valid."

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:

        dataset = Dataset.from_dict({
            'matches': df['matches'].tolist(),
            'teams': df['teams'].tolist(),
            'round': df['round'].tolist(),
            'season': df['season'].tolist(),
            'date': df['date'].tolist(),
            'text': df['text'].tolist()  # Keep the original text
        })

        return dataset

    def __len__(self):
        return len(self.train_data)
