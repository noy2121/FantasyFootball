import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Dict, List

# Constants
MODEL_NAME = "gpt2"  # Replace with your preferred model
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01


def load_system_prompts() -> Dict[str, str]:
    """Load system prompts from files."""
    with open('system_prompt_1.txt', 'r') as f:
        task_instructions = f.read()
    with open('system_prompt_2.txt', 'r') as f:
        league_rules = f.read()
    return {
        "task_instructions": task_instructions,
        "league_rules": league_rules
    }


def prepare_training_data(dataset: Dataset, system_prompts: Dict[str, str]) -> Dataset:
    """Prepare the training data for self-supervised learning, handling various data types."""

    def format_example(example: Dict) -> Dict:
        formatted_text = ""
        if 'source' in example:
            if example['source'] == 'clubs':
                formatted_text = f"Club: {example['club_name']}\n"
                formatted_text += f"Champions League Titles: {example['number_of_champions_league_titles']}\n"
                for year in range(2017, 2024):
                    formatted_text += f"{year}/{year + 1} CL Place: {example[f'{year}/{year + 1}_champions_league_place']}\n"
            elif example['source'] == 'player':
                formatted_text = f"Player: {example['player_name']}\n"
                formatted_text += f"Position: {example['position']}\n"
                formatted_text += f"Club: {example['club_name']}\n"
                formatted_text += f"Date of Birth: {example['date_of_birth']}\n"
                for year in range(2017, 2024):
                    formatted_text += f"{year}/{year + 1} Season:\n"
                    formatted_text += f"  Goals: {example[f'{year}/{year + 1}_total_goals']}\n"
                    formatted_text += f"  Assists: {example[f'{year}/{year + 1}_total_assists']}\n"
                    formatted_text += f"  Yellow Cards: {example[f'{year}/{year + 1}_total_yellow_cards']}\n"
                    formatted_text += f"  Red Cards: {example[f'{year}/{year + 1}_total_red_cards']}\n"
                    formatted_text += f"  Lineups: {example[f'{year}/{year + 1}_lineups']}\n"
            elif example['source'] == 'events':
                formatted_text = f"Event Type: {example['event_type']}\n"
                formatted_text += f"Match: {example['match']}\n"
                formatted_text += f"Club: {example['club_name']}\n"
                formatted_text += f"Date: {example['date']}\n"
                formatted_text += f"Minute: {example['minute']}\n"
                formatted_text += f"Player: {example['player_name']}\n"
            elif example['source'] == 'games':
                formatted_text = f"Competition: {example['competition_type']}\n"
                formatted_text += f"Date: {example['date']}\n"
                formatted_text += f"Home Club: {example['home_club_name']}\n"
                formatted_text += f"Away Club: {example['away_club_name']}\n"
                formatted_text += f"Score: {example['home_club_goals']} - {example['away_club_goals']}\n"
                formatted_text += f"Home Formation: {example['home_club_formation']}\n"
                formatted_text += f"Away Formation: {example['away_club_formation']}\n"
                if 'aggregate' in example:
                    formatted_text += f"Aggregate: {example['aggregate']}\n"

        formatted_text += "Analysis:\n"  # The model will learn to generate analysis
        return {"text": formatted_text}

    formatted_dataset = dataset.map(format_example)

    # Add system prompts at the beginning of the dataset
    system_prompt_text = f"{system_prompts['task_instructions']}\n\n{system_prompts['league_rules']}\n\n"
    formatted_dataset = concatenate_datasets([
        Dataset.from_dict({"text": [system_prompt_text]}),
        formatted_dataset
    ])

    return formatted_dataset


def create_few_shot_examples(system_prompts: Dict[str, str]) -> List[Dict]:
    """Create concise few-shot learning examples."""
    examples = []
    for _ in range(3):  # Reduce to 3 examples to manage length
        example = {
            "input": f"Available Players:\n"
                     "[Concise list of 15 key players with essential stats]\n\n"
                     "Create a fantasy team based on the given rules and available players.",
            "output": "Team Analysis:\n"
                      "[Brief analysis of team composition]\n\n"
                      "Selected Team:\n"
                      "[List of 11 players with positions]\n"
                      "Total cost: X million\n"
                      "Key Selection Rationale:\n"
                      "[Brief explanation for 2-3 key selections]"
        }
        examples.append(example)
    return examples


def apply_few_shot_learning(model, tokenizer, few_shot_examples, system_prompts: Dict[str, str]):
    """Apply few-shot learning with attention to prompt length."""
    system_text = f"{system_prompts['task_instructions']}\n\n{system_prompts['league_rules']}\n\n"

    for example in few_shot_examples:
        full_prompt = system_text + example["input"]
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

        # Check if the input is too long
        if inputs['input_ids'].shape[1] == MAX_LENGTH:
            print("Warning: Input was truncated. Consider shortening the system prompts or example.")

        outputs = model.generate(**inputs, max_length=MAX_LENGTH * 2)
        print(f"Input: {example['input']}")
        print(f"Generated output: {tokenizer.decode(outputs[0])}")
        print(f"Expected output: {example['output']}")
        print("---")


# ... [rest of the code remains the same] ...

def main():
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load your custom dataset
    fantasy_dataset = FantasyDataset("path/to/your/data", MODEL_NAME, MAX_LENGTH)

    # Load system prompts
    system_prompts = load_system_prompts()

    # Prepare training data
    train_dataset = prepare_training_data(fantasy_dataset.train_dataset, system_prompts)
    eval_dataset = prepare_training_data(fantasy_dataset.test_dataset, system_prompts)

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(model, tokenizer, train_dataset, eval_dataset)

    # Create few-shot examples
    few_shot_examples = create_few_shot_examples(system_prompts)

    # Apply few-shot learning
    apply_few_shot_learning(fine_tuned_model, tokenizer, few_shot_examples, system_prompts)

    # Save the final model
    fine_tuned_model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")


if __name__ == "__main__":
    main()