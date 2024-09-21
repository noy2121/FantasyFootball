import os
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import re
import hydra
import torch
torch.cuda.empty_cache()

from typing import List, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
from src.model.fantasy_model import FantasyModel
from src.model.rag.fantasy_rag import SeasonSpecificRAG

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo.eval_frame")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")


def get_user_input():
    sample_user_prompt = (
        "matches: [Copenhagen vs Manchester City, "
        "RB Leipzig vs Real Madrid, "
        "Paris Saint-Germain vs Real Sociedad, "
        "Lazio vs Bayern Munich, "
        "PSV Eindhoven vs Borussia Dortmund, "
        "Inter Milan vs Atletico Madrid, "
        "Porto vs Arsenal, "
        "Napoli vs Barcelona]\n"
        "round: Group Stage\n"
        "season: 2022/23\n"
        "date: 2022-11-23")

    return sample_user_prompt


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
    print('')
    #print("\nOverrides:")
    #print(OmegaConf.to_yaml(cfg.overrides))

    mode = cfg.mode
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "fine_tune":
        model = FantasyModel(cfg, device)
        model.fine_tune()

    elif mode == "evaluate":
        # TODO: Implement evaluation logic
        print("Evaluation mode not yet implemented")

    elif mode == "inference":
        model_dir = cfg.inference.model_dir
        model = FantasyModel.load_from_checkpoint(model_dir, device)
        user_prompt = get_user_input()
        result = model.inference(user_prompt)
        print("\nInference Result:")
        print_team(result)

    elif mode == "build_rag":
        print("Start building RAG dataset")
        rag_system = SeasonSpecificRAG(cfg.rag)
        rag_system.prepare_rag_data()
        rag_system.build_indices()
        rag_system.save()

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
