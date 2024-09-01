import os
import json
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple

from datasets import Dataset, DatasetDict


class FantasyDataset:
    def __init__(self, cfg, max_length=512):
        self.data_dir = cfg.data_dir
        self.train_filepath = cfg.train_filepath
        self.test_filepath = cfg.test_filepath
        self.max_length = max_length

        self.matches_per_round = {
            "final": 1,
            "semi-final": 2,
            "quarter-final": 4,
            "round of 16": 8,
            "group stage": 16
        }

        # Load train and test data
        self.train_data, self.test_data = self.load_data()

        # Create a DatasetDict to hold both datasets
        self.dataset_dict = DatasetDict({
            'train': self.prepare_dataset(self.train_data),
            'test': self.prepare_dataset(self.test_data)
        })

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print('Load data for training')
        train_data = self._load_json_file(f'{self.train_filepath}/train.json')
        test_data = self._load_json_file(f'{self.test_filepath}/test.json')
        return train_data, test_data

    def _load_json_file(self, filename: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, filename)
        data = []
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for sample_id, sample in json_data.items():
                parsed_sample = self._process_prompt(sample_id, sample)
                parsed_sample['sample_id'] = sample_id
                data.append(parsed_sample)
            return pd.DataFrame(data)

    @staticmethod
    def parse_prompt(prompt: str) -> Tuple[List[Tuple[str, str]], str, str, str, List[str]]:

        matches = prompt.split("matches: [")[1].split("]")[0]
        matches = matches.split(", ")
        matches = [tuple(match.split(' vs ')) for match in matches]
        kn_round = re.search(r'round: ([\w\s-]+)(?=\n|$)', prompt)
        season = re.search(r'season: (\d{4}[-/]\d{2,4})', prompt)
        date_str = re.search(r'date: (\d{4}-\d{2}-\d{2})', prompt)
        teams = [team for match in matches for team in match]

        return matches, kn_round.group(1), season.group(1), date_str.group(1), teams

    def _process_prompt(self, sample_id, prompt: str) -> Dict[str, List[str]]:

        matches, kn_round, season, date_str, teams = self.parse_prompt(prompt)
        valid, msg = self.validate_sample(matches, kn_round, date_str)
        if not valid:
            raise ValueError(f'{msg}, {sample_id=}')

        return {
            'matches': matches,
            'teams': teams,
            'round': kn_round,
            'season': season,
            'date': date_str,
            'text': prompt  # Keep the original text for tokenization
        }

    def validate_sample(self, matches: List[Tuple[str, str]], kn_round: str, date_str: str):
        knockout_round = kn_round.lower().strip()

        if knockout_round not in self.matches_per_round:
            return False, f"Invalid round: {knockout_round}. Expected one of {', '.join(self.matches_per_round.keys())}."

        if len(matches) != self.matches_per_round[knockout_round]:
            return False, f"Number of matches ({len(matches)}) does not match the expected number for {knockout_round} ({self.matches_per_round[knockout_round]})."

        # Validate number of teams
        teams = set(team for match in matches for team in match)
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
