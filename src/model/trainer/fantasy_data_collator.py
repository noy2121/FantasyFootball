from typing import Dict, List, Any
import torch

from model.rag.fantasy_rag import SeasonSpecificRAG
from system_prompts import instruction_prompt, full_rules_prompt


class FantasyTeamDataCollator:
    def __init__(self, tokenizer, rag_retriever: SeasonSpecificRAG, max_length: int, eval_steps: int):

        self.tokenizer = tokenizer
        self.rag_retriever = rag_retriever
        self.max_length = max_length
        self.eval_steps = eval_steps
        self.steps = 0

    def __call__(self, batch):

        teams_batch = [sample['teams'] for sample in batch]
        dates_batch = [sample['date'] for sample in batch]
        seasons_batch = [sample['season'] for sample in batch]

        rag_info_batch = self.rag_retriever.retrieve_relevant_info(teams_batch, dates_batch, seasons_batch)

        processed_samples = []
        for i, sample in enumerate(batch):
            processed_samples.append(self.process_sample(sample, rag_info_batch[i]))

        processed_samples = [result for result in processed_samples if result is not None]

        if not processed_samples:
            raise ValueError("All samples in the batch failed to process")

        batch_output = self.collate_batch(processed_samples)

        return batch_output

    def process_sample(self, sample: Dict[str, Any], rag_info: Dict[str, List[str]]) -> Dict[str, Any]:

        combined_input = self.combine_input_with_rag(sample['text'], rag_info)
        input_encodings = self.tokenizer(combined_input, truncation=True,
                                         max_length=self.max_length, padding="max_length")

        return {
            "input_ids": torch.tensor(input_encodings["input_ids"]),
            "attention_mask": torch.tensor(input_encodings["attention_mask"]),
            "labels": torch.tensor(input_encodings["input_ids"]),
            "matches": sample['matches'],
            "round": sample['round']
        }

    def combine_input_with_rag(self, input_text: str, rag_info: Dict[str, List[str]]) -> str:

        combined_input = (f"{input_text}\n\n"
                          f"Relevant Information:\n"
                          f"Teams Info:{rag_info['teams']}\n"
                          f"Players Info:{rag_info['players']}")

        # add system prompts occasionally
        if self.steps % self.eval_steps == 0:
            combined_input = (f"Instructions: {instruction_prompt}\n\n"
                              f"League Rules: {full_rules_prompt}\n\n"
                              f"{combined_input}")
        self.steps += 1

        return combined_input

    @staticmethod
    def collate_batch(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "matches": [item["matches"] for item in batch],
            "round": [item["round"] for item in batch]
        }