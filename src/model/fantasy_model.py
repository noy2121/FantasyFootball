import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from rag_dataset import SeasonSpecificRAG
from fantasy_dataset import FantasyDataset
from fantasy_data_collator import FantasyTeamDataCollator
from fantasy_loss import FantasyTeamLoss


class FantasyModel:
    def __init__(self, cfg):

        self.conf = cfg
        self.data_dir = cfg.data.data_dir
        self.rag_data_dir = cfg.data.rag_dir
        self.model_name = cfg.model.model_name
        self.max_length = cfg.model.max_length
        self.out_dir = cfg.train.out_dir
        self.num_epochs = cfg.train.num_epochs
        self.bz = cfg.train.batch_size

        self.model, self.tokenizer = self.create_model_and_tokenizer()
        self.rag_retriever = SeasonSpecificRAG.load(self.rag_data_dir)
        self.fantasy_dataset = FantasyDataset(self.data_dir, self.max_length)
        self.data_collator = FantasyTeamDataCollator(self.tokenizer, self.rag_retriever, self.max_length)
        self.fantasy_team_loss = FantasyTeamLoss(self.tokenizer)

        # self.valid_team_weight = 0.9
        # self.invalid_team_weight = 1.2
        # self.quality_weight = 0.1
        # self.l2_lambda = 0.01
        # self.temperature = 0.7
        # self.consecutive_good_teams = 0
        # self.patience = 10
        # self.quality_threshold = 0.8

        self.steps = 0
        self.log_steps = 100
        self.structure_weight = 1
        self.min_structure_weight = 0.1
        self.losses = {
            'loss': [],
            'lm_loss': [],
            'structure_loss': []
        }

        self.max_players_per_team = {
            "group stage": 2,
            "round of 16": 3,
            "quarter-final": 4,
            "semi-final": 6,
            "final": 8
        }
        self.max_budget_per_round = {
            "group stage": 100,
            "round of 16": 110,
            "quarter-final": 110,
            "semi-final": 120,
            "final": 120
        }

    def create_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model, tokenizer

    def combine_with_rag(self, input_text: str, teams: List[str], date: str, season: str) -> str:
        # Retrieve RAG information
        rag_info = self.rag_retriever.retrieve_relevant_info(teams, date, season)

        # Combine input with RAG info
        combined_input = (f"{input_text}\n\n"
                          f"Relevant Information:\n"
                          f"Teams Info:{rag_info['teams']}\n"
                          f"Players Info:{rag_info['players']}")


        return combined_input

    def decode_team(self, ids) -> Tuple[Dict[str, List[Tuple[str, int]]], int]:
        decoded_text = self.tokenizer.decode(ids)
        team_info = {}
        budget_used = 0

        # Regular expressions for parsing
        team_pattern = r'Team:\n(.*?)Budget used:'
        budget_pattern = r'Budget used: (\d+)M/125M'
        position_pattern = r'\t([^:]+): (.+)'
        player_pattern = r'([^(]+)\((\d+)M\)'

        # Extract team information
        team_match = re.search(team_pattern, decoded_text, re.DOTALL)
        if team_match:
            team_text = team_match.group(1)
            for line in team_text.split('\n'):
                position_match = re.match(position_pattern, line)
                if position_match:
                    position, players = position_match.groups()
                    team_info[position] = []
                    for player in players.split(', '):
                        player_match = re.match(player_pattern, player)
                        if player_match:
                            name, cost = player_match.groups()
                            team_info[position].append((name.strip(), int(cost)))
                            budget_used += int(cost)

        # Extract budget information
        budget_match = re.search(budget_pattern, decoded_text)
        if budget_match:
            budget_used = int(budget_match.group(1))

        return team_info, budget_used

    def is_team_valid(self, team_info: Dict[str, List[Tuple[str, int]]], budget_used: int,
                      matches: List[str], knockout_round: str) -> Tuple[bool, str]:

        formation = {
            "Goalkeeper": 0,
            "Defence": 0,
            "Midfield": 0,
            "Attack": 0
        }
        team_counts = {}
        total_players = 0

        # check available players
        for position, players in team_info.items():
            formation[position] = len(players)
            total_players += len(players)

            for player, cost in players:
                # Check if player is in any of the matches
                team = next((team for match in matches for team in match.split(' vs ') if player in team), None)
                if team:
                    team_counts[team] = team_counts.get(team, 0) + 1
                else:
                    return False, f"Player {player} is not part of a team in any of the provided matches."

        # check formation
        if (formation["Goalkeeper"] != 1 or
                formation["Defence"] < 3 or formation["Defence"] > 5 or
                formation["Midfield"] < 3 or formation["Midfield"] > 5 or
                formation["Attack"] < 1 or formation["Attack"] > 3):
            return False, f"Invalid formation: {formation}"

        # check total number of players
        if total_players != 11:
            return False, f"Invalid number of players: {total_players}"

        # check number of players per team
        if knockout_round not in self.max_players_per_team:
            return False, f"Invalid round: {knockout_round}"
        for team, count in team_counts.items():
            if count > self.max_players_per_team[knockout_round]:
                return False, f"Too many players ({count}) from team {team} for {knockout_round}"

        # check budget
        if budget_used > self.max_budget_per_round[knockout_round]:
            return False, f"Team cost ({budget_used}M) exceeds budget ({self.max_budget_per_round[knockout_round]}M)"

        return True, "Team is valid"

    def estimate_team_quality(self, team: Dict[str, List[Tuple[str, int]]]) -> float:
        # TODO: Implement team quality estimation
        # This could consider factors like:
        # - Player recent performance (you might need to fetch this data)
        # - Team diversity (players from different real-world teams)
        # - Balance of positions
        # - Value for money (performance vs cost)
        pass

    def fantasy_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.labels_id
        matches = eval_pred.inputs['matches']
        knockout_rounds = eval_pred.inputs['round']
        predictions = logits.argmax(axis=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        validity_scores = []
        quality_scores = []
        for pred, match, kn_round in zip(decoded_preds, matches, knockout_rounds):
            team, budget = self.decode_team(pred)
            is_valid, _ = self.is_team_valid(team, budget, match, kn_round)
            validity_scores.append(int(is_valid))
            if is_valid:
                quality_scores.append(self.estimate_team_quality(team))

        validity_rate = sum(validity_scores) / len(validity_scores)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        combined_score = validity_rate * avg_quality

        return {
            "validity_rate": validity_rate,
            "avg_quality": avg_quality,
            "combined_score": combined_score
        }

    def _log_metrics(self):
        avg_loss = np.mean(self.losses['loss'][-self.log_steps:])
        avg_lm_loss = np.mean(self.losses['lm_loss'][-self.log_steps:])
        avg_structure_loss = np.mean(self.losses['structure_loss'][-self.log_steps:])
        print(f"Step {self.steps}: Avg Loss: {avg_loss:.4f}, "
              f"Avg LM Loss: {avg_lm_loss:.4f}, "
              f"Avg Structure Loss: {avg_structure_loss:.4f}")

    def fantasy_loss(self, outputs, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.steps += 1

        # calculate loss
        lm_loss, structure_loss = self.fantasy_team_loss(outputs.logits, batch['input_ids'])

        # combine losses with updated weight
        total_loss = lm_loss + (self.structure_weight * structure_loss)

        # add L2 regularization
        l2_lambda = 0.01  # Adjust this value as needed
        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
        total_loss += l2_lambda * l2_reg

        # update losses
        self.losses['loss'].append(total_loss.item())
        self.losses['lm_loss'].append(lm_loss.item())
        self.losses['structure_loss'].append(structure_loss.item())

        # log metrics every 1000 steps
        if self.steps % self.log_steps == 0:
            self._log_metrics()

        # decrease structure weight over time (ensure it doesn't drop below a minimum value)
        self.structure_weight = max(self.min_structure_weight, self.structure_weight * 0.9)

        return total_loss

    def fine_tune(self):
        train_dataset = self.fantasy_dataset.dataset_dict('train')
        eval_dataset = self.fantasy_dataset.dataset_dict('train')

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01,
        )

        training_args = TrainingArguments(
            output_dir=self.out_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.bz,
            per_device_eval_batch_size=self.bz,
            load_best_model_at_end=True,
            metric_for_best_model='combined_score',
            greater_is_better=True,
            evaluation_strategy='steps',
            eval_steps=self.log_steps,
            save_steps=self.log_steps,
            save_total_limit=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_loss=self.fantasy_loss,
            compute_metrics=self.fantasy_metrics,
            callbacks=[early_stopping_callback]
        )

        trainer.train()

    def inference(self, prompt: str) -> Dict[str, List[Tuple[str, int]]]:
        matches = re.findall(r'([\w\s]+) vs ([\w\s]+)', prompt)
        kn_round = re.search(r'round: ([\w\s-]+)', prompt)
        season = re.search(r'season: (\d{4}-\d{2}-\d{2})', prompt)
        date_str = re.search(r'date: (\d{4}-\d{2}-\d{2})', prompt)
        teams = [team for match in matches for team in match]
        combined_input = self.combine_with_rag(prompt, teams, date_str, season)

        inputs = self.tokenizer(combined_input, return_tensors="pt", truncation=True,
                                max_length=self.max_length)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            prefix="Team:\n"
        )

        team, budget_used = self.decode_team(outputs[0])
        return team
