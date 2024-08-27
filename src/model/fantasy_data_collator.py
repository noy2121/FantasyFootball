import re
import torch
from transformers import DataCollator

from rag_dataset import SeasonSpecificRAG
from system_prompts import instruction_prompt, short_rules_prompt, full_rules_prompt


class FantasyTeamDataCollator(DataCollator):
    def __init__(self, tokenizer, rag_retriever: SeasonSpecificRAG, max_length: int, eval_steps: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.rag_retriever = rag_retriever
        self.max_length = max_length
        self.eval_steps = eval_steps
        self.steps = 0

    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "matches": [], "round": []}
        for feature in features:

            # Assume feature['text'] contains the input prompt
            date = feature['date']
            season = feature['season']
            teams = feature['teams']
            input_text = feature['text']

            # Retrieve RAG information
            rag_info = self.rag_retriever.retrieve_relevant_info(teams, date, season)

            # Combine input with RAG info
            combined_input = (f"{input_text}\n\n"
                              f"Relevant Information:\n"
                              f"Teams Info:{rag_info['teams']}\n"
                              f"Players Info:{rag_info['players']}")

            # Decide which system prompt to use
            if self.steps % self.eval_steps == 0:
                combined_input = (f"Instructions: {instruction_prompt}\n\n"
                                  f"League Rules: {full_rules_prompt}\n\n"
                                  f"{combined_input}")

            # Tokenize combined input
            input_encodings = self.tokenizer(combined_input, truncation=True,
                                             max_length=self.max_length, padding="max_length")

            # Extract matches and knockout_round from the input text
            matches = re.search(r"matches: \[(.*?)\]", feature['text']).group(1).split(', ')
            knockout_round = re.search(r"round: (.*?)$", feature['text'], re.MULTILINE).group(1)

            batch["input_ids"].append(torch.tensor(input_encodings["input_ids"]))
            batch["attention_mask"].append(torch.tensor(input_encodings["attention_mask"]))
            batch["labels"].append(torch.tensor(input_encodings["input_ids"]))
            batch["matches"].append(matches)
            batch["round"].append(knockout_round)

        # Stack tensors, but keep matches and knockout_round as lists
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])

        self.steps += 1

        return batch
