import re
from typing import Dict, List, Tuple
import torch
import numpy as np


def last_5_games_avg_score(stats):
    avg_score = 0
    if stats['last_5']:
        arr = np.array([(d['goals'], d['assists'], d['lineups']) for d in stats['last_5']])
        goals, assists, lineups = arr.sum(axis=0)

        avg_score = ((4 * goals) + (3 * assists) + (2 * lineups)) / len(stats['last_5'])

    return np.round(avg_score, 3)


def seasonal_avg_score(player_stats, num_games):
    if num_games == 0:
        return 0
    avg_score = ((4 * player_stats['seasonal']['goals'])
                 + (3 * player_stats['seasonal']['assists'])
                 + (2 * player_stats['seasonal']['lineups'])) / num_games

    return np.round(avg_score, 3)


class FantasyMetric:
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_player_score = 100
        self.max_budget_per_round = {
            "group stage": 100,
            "round of 16": 110,
            "quarter-final": 110,
            "semi-final": 120,
            "final": 120
        }
        self.max_players_per_team = {
            "group stage": 2,
            "round of 16": 3,
            "quarter-final": 4,
            "semi-final": 6,
            "final": 8
        }

        self.metrics = {
            "validity_rate": [],
            "avg_quality": [],
            "combined_score": []
        }

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

    @staticmethod
    def estimate_player_score(player_stats: Dict, club_stats: Dict, position: str) -> float:
        """
        compute player estimated points based on past performance
        """
        if not player_stats:
            return 0.0
        player_last_5_avg_score = last_5_games_avg_score(player_stats)
        player_season_avg_score = seasonal_avg_score(player_stats, club_stats['seasonal']['games'])
        position_multiplier = {
            'Goalkeeper': 3,
            'Defender': 2,
            'Midfielder': 1.4,
            'Forward': 1.0
        }.get(position, 1.0)

        player_score = (player_last_5_avg_score + player_season_avg_score) * position_multiplier

        # add team performance to player's score
        result_map = {'win': 1, 'tie': 0, 'lose': -1}
        last_5_cl = [result_map[res] for res in club_stats['last_5_cl']]
        last_5_dl = [result_map[res] for res in club_stats['last_5_dl']]
        player_team_score = (sum(last_5_cl) / min(1, len(club_stats['last_5_cl']))
                             + sum(last_5_dl) / min(1, len(club_stats['last_5_dl'])))

        # Calculate final score and add random points
        final_score = player_score + player_team_score + np.random.randint(-5, 5)
        return max(1, final_score)

    def estimate_team_quality(self, team: Dict[str, List[Tuple[str, int]]], date: str, budget_used: int, kn_round: str) -> float:
        """
        team = {'goalkeeper': [(player_name, cost)...],'defence': [(player_name, cost)...],...}
        """

        score = 0.0
        for position, players in team.items():
            for player_name, _ in players:
                player_stats, club_stats = self.data_stats_cache.get(date, player_name)
                player_score = self.estimate_player_score(player_stats, club_stats, position)
                player_score = min(player_score / self.max_player_score, 1.0)  # normalize between 0, 1
                score += player_score

        # reward if the model use the budget correctly
        if budget_used > self.max_budget_per_round[kn_round] - 5:
            score = min(1.2 * score, 1.0)
        elif budget_used > self.max_budget_per_round[kn_round] - 10:
            score = min(1.05 * score, 1.0)

        return score

    def decode_team(self, decoded_text) -> Tuple[Dict[str, List[Tuple[str, int]]], int]:
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

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        # matches = eval_pred.inputs['matches']
        # knockout_rounds = eval_pred.inputs['round']
        # dates = eval_pred.inputs['date']

        matches = self.eval_dataset['matches']
        knockout_rounds = self.eval_dataset['round']
        dates = self.eval_dataset['date']

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

        self.metrics["validity_rate"].append(validity_rate)
        self.metrics["avg_quality"].append(avg_quality)
        self.metrics["combined_score"].append(combined_score)
        self.log_metrics()

        return {
            "validity_rate": validity_rate,
            "avg_quality": avg_quality,
            "combined_score": combined_score
        }

    def log_metrics(self):
        avg_score = np.mean(self.metrics['combined_score'])
        avg_quality = np.mean(self.metrics['avg_quality'])
        avg_vr = np.mean(self.metrics['validity_rate'])
        print(f"Evaluation: Avg Score: {avg_score:.4f}, "
              f"Avg Team Quality: {avg_quality:.4f}, "
              f"Avg Team Validity: {avg_vr:.4f}")

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids
