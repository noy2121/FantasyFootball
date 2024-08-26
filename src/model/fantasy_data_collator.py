import re
import torch
from transformers import DataCollator

from rag_dataset import SeasonSpecificRAG


class FantasyTeamDataCollator(DataCollator):
    def __init__(self, tokenizer, rag_retriever: SeasonSpecificRAG, max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.rag_retriever = rag_retriever
        self.max_length = max_length

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

        return batch
