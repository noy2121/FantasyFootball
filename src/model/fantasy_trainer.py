import torch
import numpy as np

from transformers import Trainer
from typing import Dict, Union, Any


class FantasyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract custom arguments
        self.fantasy_team_loss = kwargs.pop('fantasy_team_loss', None)
        self.eval_steps = kwargs.pop('eval_steps', 100)
        self.structure_weight = kwargs.pop('initial_structure_weight', 1.0)
        self.min_structure_weight = kwargs.pop('min_structure_weight', 0.1)

        # Initialize Trainer with remaining arguments
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.losses = {
            'loss': [],
            'lm_loss': [],
            'structure_loss': []
        }

    def compute_loss(self, model, inputs, return_outputs=False):

        model_inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        outputs = model(**model_inputs)

        # Calculate custom loss
        lm_loss, structure_loss = self.fantasy_team_loss(outputs.logits, inputs['input_ids'])

        # Combine losses with updated weight
        total_loss = lm_loss + (self.structure_weight * structure_loss)

        # Add L2 regularization
        l2_lambda = 0.01  # Adjust this value as needed
        l2_reg = torch.sum(torch.stack([p.pow(2.0).sum() for p in model.parameters()]))
        total_loss += l2_lambda * l2_reg

        # Update losses
        self.losses['loss'].append(total_loss.item())
        self.losses['lm_loss'].append(lm_loss.item())
        self.losses['structure_loss'].append(structure_loss.item())

        # Log metrics every eval_steps
        if self.steps % self.eval_steps == 0:
            self._log_metrics()

        # Decrease structure weight over time
        self.structure_weight = np.maximum(self.min_structure_weight, self.structure_weight * 0.9)

        self.steps += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _move_model_to_device(self, model, device):
        pass

    def _log_metrics(self):
        avg_loss = np.mean(self.losses['loss'][-self.eval_steps:])
        avg_lm_loss = np.mean(self.losses['lm_loss'][-self.eval_steps:])
        avg_structure_loss = np.mean(self.losses['structure_loss'][-self.eval_steps:])
        print(f"Step {self.steps}: Avg Loss: {avg_loss:.4f}, "
              f"Avg LM Loss: {avg_lm_loss:.4f}, "
              f"Avg Structure Loss: {avg_structure_loss:.4f}")

    def train(self, resume_from_checkpoint: Union[str, bool] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None, **kwargs):
        # Reset steps and losses before training
        self.steps = 0
        self.losses = {k: [] for k in self.losses}
        return super().train(resume_from_checkpoint, trial, **kwargs)