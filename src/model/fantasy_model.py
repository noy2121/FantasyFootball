import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from peft import PeftModel, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LoRA_MODULES_MAPPING

from rag_dataset import SeasonSpecificRAG
from fantasy_dataset import FantasyDataset
from fantasy_data_collator import FantasyTeamDataCollator
from fantasy_loss import FantasyTeamLoss
from fantasy_stats import DataStatsCache


class FantasyModel:
    def __init__(self, cfg):

        self.conf = cfg
        self.data_dir = cfg.data.data_dir
        self.rag_data_dir = cfg.rag.rag_dir
        self.model_name = cfg.model.model_name
        self.max_length = cfg.model.max_length
        self.peft_method = cfg.train.peft_method
        self.num_epochs = cfg.train.num_epochs
        self.bz = cfg.train.batch_size
        self.out_dir = f"{cfg.train.out_dir}/{self.model_name.split('/')[-1]}/{self.peft_method}"
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.steps = 0
        self.eval_steps = cfg.train.evaluation_steps
        self.structure_weight = 1
        self.min_structure_weight = 0.1
        self.losses = {
            'loss': [],
            'lm_loss': [],
            'structure_loss': []
        }

        self.model, self.tokenizer = self.create_model_and_tokenizer()
        self.rag_retriever = SeasonSpecificRAG.load(self.rag_data_dir)
        self.fantasy_dataset = FantasyDataset(self.data_dir, self.max_length)
        self.data_collator = FantasyTeamDataCollator(self.tokenizer, self.rag_retriever, self.max_length, self.eval_steps)
        self.fantasy_team_loss = FantasyTeamLoss(self.tokenizer)
        self.data_stats_cache = DataStatsCache(self.conf.rag.estimation_data_dir)

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

        self.max_player_score = 100

        if self.peft_method != 'all':
            self.model = self.apply_peft_model()

    def create_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model, tokenizer

    def apply_peft_model(self):
        if self.peft_method == 'lora':
            default_target_modules = LoRA_MODULES_MAPPING.get(self.model_name, ["q_proj", "v_proj"])
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.conf.peft.r,
                lora_alpha=self.conf.peft.lora_alpha,
                lora_dropout=self.conf.peft.dropout,
                bias='none',
                target_modules=default_target_modules
            )
        elif self.peft_method == 'prefix_tuning':
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                prefix_projection=True
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")

        model = get_peft_model(self.model, peft_config)
        return model

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

    def _estimate_player_score(self, player_stats: Dict, club_stats: Dict, position: str) -> float:
        """
        compute player estimated points based on past performance
        """
        if not player_stats:
            return 0.0
        goals, assists, lineups = 0, 0, 0
        if player_stats['last_5']:
            arr = np.array([(d['goals'], d['assists'], d['lineups']) for d in player_stats['last_5']])
            goals, assists, lineups = arr.sum(axis=0)

        player_last_5_score = ((4 * goals) + (3 * assists) + (1 * lineups)) / len(goals)
        player_season_score = ((4 * player_stats['season']['goals'])
                               + (3 * player_stats['season']['assists'])
                               + (1 * player_stats['season']['lineups'])) / club_stats['seasonal']['games']

        position_multiplier = {
            'Goalkeeper': 3,
            'Defender': 2,
            'Midfielder': 1.4,
            'Forward': 1.0
        }.get(position, 1.0)

        player_score = (player_last_5_score + player_season_score) * position_multiplier

        # add team performance to player's score
        result_map = {'win': 1, 'tie': 0, 'lose': -1}
        last_5_cl = [result_map[res] for res in club_stats['last_5_cl']]
        last_5_dl = [result_map[res] for res in club_stats['last_5_dl']]
        player_team_score = (sum(last_5_cl) / min(1, len(club_stats['last_5_cl']))
                             + sum(last_5_dl) / min(1, len(club_stats['last_5_dl'])))

        # Calculate final score and normalize to be between 0 and 1
        final_score = player_score + player_team_score
        return min(final_score / self.max_player_score, 1.0)

    def estimate_team_quality(self, team: Dict[str, List[Tuple[str, int]]], date: str, budget_used: int, kn_round: str) -> float:
        """
        team = {'goalkeeper': [(player_name, cost)...],'defence': [(player_name, cost)...],...}
        """

        score = 0.0
        for position, players in team.items():
            for player_name, _ in players:
                player_stats, club_stats = self.data_stats_cache.get(date, player_name)
                player_score = self._estimate_player_score(player_stats, club_stats, position)
                score += player_score

        # reward if the model use the budget correctly
        if budget_used > self.max_budget_per_round[kn_round] - 5:
            score = min(1.1 * score, 1.0)
        elif budget_used > self.max_budget_per_round[kn_round] - 10:
            score = min(1.05 * score, 1.0)

        return score

    def fantasy_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.labels_id
        matches = eval_pred.inputs['matches']
        knockout_rounds = eval_pred.inputs['round']
        dates = eval_pred.inputs['date']
        predictions = logits.argmax(axis=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        validity_scores = []
        quality_scores = []
        for pred, match, kn_round, date in zip(decoded_preds, matches, knockout_rounds, dates):
            team, budget_used = self.decode_team(pred)
            is_valid, _ = self.is_team_valid(team, budget_used, match, kn_round)
            validity_scores.append(int(is_valid))
            if is_valid:
                quality_scores.append(self.estimate_team_quality(team, date, budget_used, kn_round))

        validity_rate = sum(validity_scores) / len(validity_scores)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        combined_score = validity_rate * avg_quality

        return {
            "validity_rate": validity_rate,
            "avg_quality": avg_quality,
            "combined_score": combined_score
        }

    def _log_metrics(self):
        avg_loss = np.mean(self.losses['loss'][-self.eval_steps:])
        avg_lm_loss = np.mean(self.losses['lm_loss'][-self.eval_steps:])
        avg_structure_loss = np.mean(self.losses['structure_loss'][-self.eval_steps:])
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
        if self.steps % self.eval_steps == 0:
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
            eval_steps=self.eval_steps,
            save_steps=self.eval_steps,
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

        # Save the fine-tuned model
        self.save_model()

    def inference(self, prompt: str) -> Dict[str, List[Tuple[str, int]]]:
        matches, kn_round, season, date_str, teams = self.fantasy_dataset.parse_prompt(prompt)

        if not self.conf.inference.vanilla:
            prompt = self.combine_with_rag(prompt, teams, date_str, season)

        if self.peft_method == 'lora' and not hasattr(self, 'merged_model'):
            merged_model = self.model.merge_and_unload()
            model_for_inference = merged_model
        else:
            model_for_inference = self.model

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_length)
        outputs = model_for_inference.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=5,
            temperature=0.7,
            prefix="Team:\n"
        )

        team, budget_used = self.decode_team(outputs[0])
        return team

    @classmethod
    def load_from_checkpoint(cls, model_dir):
        config = torch.load(f"{model_dir}/config.pt")

        # Create an instance of FantasyModel
        instance = cls(config)

        # Load the tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")

        # Load the PEFT method used
        with open(f"{model_dir}/peft_method.txt", "r") as f:
            peft_method = f.read().strip()

        # Load the model
        if peft_method == 'all':
            model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/model")
        else:
            # Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/base_model")
            # Load the PEFT adapters
            model = PeftModel.from_pretrained(base_model, f"{model_dir}/peft_model")

        instance.model = model
        instance.peft_method = peft_method

        return instance

    def save_model(self):
        # Save the configuration
        torch.save(self.conf, f"{self.out_dir}/config.pt")

        # Save the tokenizer
        self.tokenizer.save_pretrained(f"{self.out_dir}/tokenizer")

        # Save the model
        if self.peft_method == 'all':
            # For full fine-tuning, save the entire model
            self.model.save_pretrained(f"{self.out_dir}/model")

        elif isinstance(self.model, PeftModel):
            # For PEFT methods (LoRA, Prefix Tuning, etc.)
            # Save the base model
            self.model.base_model.save_pretrained(f"{self.out_dir}/base_model")
            # Save the PEFT adapters
            self.model.save_pretrained(f"{self.out_dir}/peft_model")
        else:
            raise ValueError(f"Unexpected model type for peft_method: {self.peft_method}")

        # Save the PEFT method used
        with open(f"{self.out_dir}/peft_method.txt", "w") as f:
            f.write(self.peft_method)

        print(f"Checkpoint saved to {self.out_dir}")
