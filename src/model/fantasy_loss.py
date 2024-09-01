import re
from typing import Tuple
import torch
import torch.nn.functional as F


class FantasyTeamLoss(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standard language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                  ignore_index=-100)

        # Generate team from logits
        generated_ids = torch.argmax(logits, dim=-1)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        structure_loss = torch.sum(torch.tensor(
            [self.structure_loss(text) for text in generated_texts],
            device=logits.device
        ))

        return lm_loss, structure_loss

    @staticmethod
    def structure_loss(text: str) -> int:
        expected_structure = [
            r"Team:",
            r"\tGoalkeeper: .+ \(\d+M\)",
            r"\tDefence: (?:.+ \(\d+M\),? ?){3,5}",
            r"\tMidfield: (?:.+ \(\d+M\),? ?){3,5}",
            r"\tAttack: (?:.+ \(\d+M\),? ?){1,3}",
            r"Budget used: \d+M/\d+M"
        ]

        loss = 0
        for pattern in expected_structure:
            if not re.search(pattern, text):
                loss += 1
        return loss
