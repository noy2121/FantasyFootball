import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
from omegaconf import DictConfig

from src.evaluation.visualization import FantasyVisualizer
from src.model.fantasy_model import FantasyModel
from src.model.fantasy_dataset import FantasyDataset
from src.model.fantasy_data_collator import FantasyTeamDataCollator
from src.model.rag_dataset import SeasonSpecificRAG


class FantasyEvaluator:
    def __init__(self, cfg: DictConfig, device):
        self.cfg = cfg
        self.device = device
        self.models = self.load_models()
        self.dataset, self.data_collator = self.prepare_data()
        self.visualizer = FantasyVisualizer() if self.cfg.evaluation.visualize else None

    def load_models(self):
        models = []
        for model_dir in self.cfg.evaluation.model_paths:
            model = FantasyModel.load_from_checkpoint(model_dir)
            model.eval()
            model.to(self.device)
            models.append(model)
        return models

    def prepare_data(self):
        dataset = FantasyDataset(self.cfg.data.data_dir, self.cfg.model.max_length)
        rag_retriever = SeasonSpecificRAG.load(self.cfg.rag.rag_dir)
        data_collator = FantasyTeamDataCollator(dataset.tokenizer, rag_retriever, self.cfg.model.max_length,
                                                self.cfg.train.eval_steps)
        return dataset, data_collator

    def evaluate_model(self, model):
        all_predictions = []
        all_labels = []
        all_matches = []
        all_rounds = []
        all_dates = []
        budget_utilization = []
        player_selections = defaultdict(int)
        round_performances = defaultdict(list)

        test_data = self.dataset.dataset_dict['test']
        for i in tqdm(range(len(test_data)), desc="Evaluating"):
            batch = self.data_collator([test_data[i]])

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=model.max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    prefix="Team:\n"
                )

            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_matches.extend(batch['matches'])
            all_rounds.extend(batch['round'])
            all_dates.extend(batch['date'])

            # Budget utilization analysis
            team, budget_used = model.decode_team(outputs[0])
            budget_utilization.append(budget_used)

            # Player selection frequency
            for position, players in team.items():
                for player, _ in players:
                    player_selections[player] += 1

            # Performance consistency across rounds
            is_valid, _ = model.is_team_valid(team, budget_used, batch['matches'][0], batch['round'][0])
            round_performances[batch['round'][0]].append(int(is_valid))

        metrics = model.fantasy_metrics({
            'predictions': all_predictions,
            'labels_id': all_labels,
            'inputs': {
                'matches': all_matches,
                'round': all_rounds,
                'date': all_dates
            }
        })

        # Additional metrics
        metrics['avg_budget_utilization'] = np.mean(budget_utilization)
        metrics['budget_utilization_std'] = np.std(budget_utilization)
        metrics['top_10_players'] = sorted(player_selections.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics['round_performance'] = {round: np.mean(perfs) for round, perfs in round_performances.items()}

        return metrics

    def time_series_analysis(self, model):
        test_data = self.dataset.dataset_dict['test']
        dates = [item['date'] for item in test_data]
        performances = []

        for i in tqdm(range(len(test_data)), desc="Time Series Analysis"):
            batch = self.data_collator([test_data[i]])

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=model.max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    prefix="Team:\n"
                )

            team, budget_used = model.decode_team(outputs[0])
            is_valid, _ = model.is_team_valid(team, budget_used, batch['matches'][0], batch['round'][0])
            performances.append(int(is_valid))

        df = pd.DataFrame({'date': dates, 'performance': performances})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['rolling_avg'] = df['performance'].rolling(window=7).mean()

        return df

    def visualize(self, metrics_list, time_series_data_list, model_names):
        if self.visualizer:
            output_dir = self.cfg.evaluation.visualization_output_dir
            os.makedirs(output_dir, exist_ok=True)
            self.visualizer.visualize_all(metrics_list, time_series_data_list, model_names, output_dir)
        else:
            print("Visualization is disabled. Set 'visualize: true' in the config to enable.")

    def run_evaluation(self):
        metrics_list = []
        time_series_data_list = []
        model_names = []

        for i, model in enumerate(self.models):
            print(f"\nEvaluating Model {i + 1}:")
            metrics = self.evaluate_model(model)
            ts_data = self.time_series_analysis(model)

            metrics_list.append(metrics)
            time_series_data_list.append(ts_data)
            model_names.append(f"Model {i + 1}")

            # Print metrics (as before)
            print(f"Validity Rate: {metrics['validity_rate']:.4f}")
            print(f"Average Quality: {metrics['avg_quality']:.4f}")
            print(f"Combined Score: {metrics['combined_score']:.4f}")
            print(f"Average Budget Utilization: {metrics['avg_budget_utilization']:.2f}M")
            print(f"Budget Utilization Std Dev: {metrics['budget_utilization_std']:.2f}M")
            print("\nTop 10 Most Selected Players:")
            for player, count in metrics['top_10_players']:
                print(f"  {player}: {count} times")
            print("\nPerformance by Round:")
            for round, perf in metrics['round_performance'].items():
                print(f"  {round}: {perf:.4f}")

            print("\nTime Series Analysis:")
            print(ts_data.describe())

        if self.cfg.evaluation.visualize:
            self.visualize(metrics_list, time_series_data_list, model_names)
