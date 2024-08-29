import os
import re
import hydra
import torch
from typing import List, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
from src.model.fantasy_model import FantasyModel
from src.model.fantasy_rag import SeasonSpecificRAG


def get_user_input():
    print("Please enter your prompt in the following format:")
    print("matches: [Team1 vs Team2, Team3 vs Team4, ...]\n"
          "round: [Group Stage/Round of 16/Quarter-final/Semi-final/Final]\n"
          "season: YYYY/YY\n"
          "date: YYYY-MM-DD")

    i = 0
    while i < 5:
        user_input = input("\nEnter your prompt: ")
        if validate_input(user_input):
            return user_input
        print("Invalid input. Please try again.")
        i += 1
    raise TimeoutError('You ran out of time. Please try later.')


def validate_input(input_str):
    # Check for matches
    if not re.search(r'matches: \[.+\]', input_str):
        return False

    # Check for round
    if not re.search(r'round: (Group Stage|Round of 16|Quarter-final|Semi-final|Final)', input_str):
        return False

    # Check for season
    if not re.search(r'season: \d{4}/\d{2}', input_str):
        return False

    # Check for date
    if not re.search(r'date: \d{4}-\d{2}-\d{2}', input_str):
        return False

    return True


def print_team(team: Dict[str, List[Tuple[str, int]]]):
    for position, players in team.items():
        print(f"{position}:")
        for player, cost in players:
            print(f"  - {player} ({cost}M)")

    total_cost = sum(cost for players in team.values() for _, cost in players)
    print(f"\nTotal Cost: {total_cost}M")


@hydra.main(config_path="src/config", config_name="conf")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    #print("\nOverrides:")
    #print(OmegaConf.to_yaml(cfg.overrides))

    mode = cfg.mode
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "fine_tune":
        model = FantasyModel(cfg)
        model.fine_tune()

    elif mode == "evaluate":
        # TODO: Implement evaluation logic
        print("Evaluation mode not yet implemented")

    elif mode == "inference":
        model = FantasyModel(cfg)
        user_prompt = get_user_input()
        result = model.inference(user_prompt)
        print("\nInference Result:")
        print_team(result)

    elif mode == "build_rag":
        print("Start building RAG dataset")
        rag_system = SeasonSpecificRAG(cfg.rag, device)
        rag_system.prepare_rag_data()
        rag_system.build_indices()
        rag_system.save(cfg.data.rag_dir)

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
