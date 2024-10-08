import os
import sys

import torch
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import EarlyStoppingCallback, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, prepare_model_for_kbit_training
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LoRA_MODULES_MAPPING

from .trainer.fantasy_trainer import FantasyTrainer
from .trainer.fantasy_dataset import FantasyDataset
from .trainer.fantasy_data_collator import FantasyTeamDataCollator
from .trainer.fantasy_loss import FantasyTeamLoss
from .rag.fantasy_rag import SeasonSpecificRAG
from .metrics.fantasy_metrics import FantasyMetric
from .fantasy_stats import DataStatsCache
from .flash_attention import apply_flash_attention
from ..system_prompts import instruction_prompt, full_rules_prompt, short_rules_prompt
from ..utils.utils import get_hftoken


class FantasyModel:
    def __init__(self, cfg, device):

        self.conf = cfg
        self.device = device
        self.data_dir = cfg.data.data_dir
        self.rag_data_dir = cfg.rag.rag_dir
        self.model_name = cfg.model.model_name
        self.max_length = cfg.model.max_length
        self.use_flash_attention = cfg.model.use_flash_attention
        self.peft_method = cfg.train.peft_method
        self.num_epochs = cfg.train.num_epochs
        self.bz = cfg.train.batch_size
        self.out_dir = f"{cfg.train.out_dir}/{self.model_name.split('/')[-1]}/{self.peft_method}"
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.eval_steps = cfg.train.evaluation_steps
        self.structure_weight = 1
        self.min_structure_weight = 0.1

        self.model, self.tokenizer = self.create_model_and_tokenizer()
        self.rag_retriever = SeasonSpecificRAG.load(self.rag_data_dir)
        self.fantasy_dataset = FantasyDataset(self.conf.data, self.max_length)
        self.data_collator = FantasyTeamDataCollator(self.tokenizer, self.rag_retriever, self.max_length, self.eval_steps)
        self.fantasy_team_loss = FantasyTeamLoss(self.tokenizer)
        self.data_stats_cache = DataStatsCache(self.conf.rag.estimation_data_dir)
        self.fantasy_metric = FantasyMetric(self.tokenizer, self.fantasy_dataset.dataset_dict['test'])

        if self.peft_method != 'all':
            self.model = self.apply_peft_model()

        if self.use_flash_attention:
            # torch.backends.cuda.enable_flash_sdp(True)
            self.model = self._apply_flash_attn()

    def create_model_and_tokenizer(self):
        print(f'Load model and tokenizer: {self.model_name}')
        hf_token = get_hftoken(self.conf.model.hf_token_filepath)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # load model
        if self.conf.model.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                use_cache=False)
            model.gradient_checkpointing_enable()

        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, token=hf_token,
                                                         trust_remote_code=True, use_cache=False)

            model.gradient_checkpointing_enable()

        return model, tokenizer

    def _apply_flash_attn(self):
        print('Using Flash Attention...')
        return apply_flash_attention(self.model)

    def apply_peft_model(self):
        print(f'Apply PEFT method [{self.peft_method}] to the model...')
        if self.peft_method == 'lora':
            default_target_modules = list(self.conf.peft.target_modules)
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
        return self.fantasy_metric.decode_team(ids)

    def is_team_valid(self, team_info: Dict[str, List[Tuple[str, int]]], budget_used: int,
                      matches: List[str], knockout_round: str) -> Tuple[bool, str]:
        return self.fantasy_metric.is_team_valid(team_info, budget_used, matches, knockout_round)

    def preprocess_logits_for_metrics(self, logits, labels):
        return self.fantasy_metric.preprocess_logits_for_metrics(logits, labels)

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        return self.fantasy_metric.compute_metrics(eval_pred)

    def fine_tune(self):

        train_dataset = self.fantasy_dataset.dataset_dict['train']
        eval_dataset = self.fantasy_dataset.dataset_dict['test']

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01,
        )

        training_args = TrainingArguments(
            output_dir=self.out_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.bz,
            per_device_eval_batch_size=self.bz,
            gradient_accumulation_steps=self.conf.train.accumulation_steps,
            load_best_model_at_end=True,
            metric_for_best_model='combined_score',
            greater_is_better=True,
            evaluation_strategy='steps',
            eval_steps=self.eval_steps // self.conf.train.accumulation_steps,
            save_steps=self.eval_steps // self.conf.train.accumulation_steps,
            save_strategy='steps',
            save_total_limit=10,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            gradient_checkpointing=True
        )

        print('\nBegin fine-tuning the model')
        trainer = FantasyTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            callbacks=[early_stopping_callback],
            fantasy_team_loss=self.fantasy_team_loss,
            initial_structure_weight=self.structure_weight,
            min_structure_weight=self.min_structure_weight
        )

        print(f"Evaluation strategy: {training_args.evaluation_strategy}")
        print(f"Evaluation steps: {training_args.eval_steps}")

        trainer.train()

        # Save the fine-tuned model
        self.save_model()

    def inference(self, prompt: str) -> Dict[str, List[Tuple[str, int]]]:
        print('Begin inference')
        matches, kn_round, season, date_str, teams = self.fantasy_dataset.parse_prompt(prompt)

        if not self.conf.inference.vanilla:
            prompt = self.combine_with_rag(prompt, teams, date_str, season)

        if self.peft_method == 'lora' and not hasattr(self, 'merged_model'):
            merged_model = self.model.merge_and_unload()
            model_for_inference = merged_model
        else:
            model_for_inference = self.model

        # add instructions
        prompt = (f"Instructions: {instruction_prompt}\n\n"
                  f"League Rules: {short_rules_prompt}\n\n"
                  f"{prompt}")

        suffix = "\nTeam:\n"
        prompt = prompt + suffix
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_length)

        # model_for_inference.to(self.device)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = model_for_inference.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=5
        )
        decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        team, budget_used = self.decode_team(decoded_preds)
        return team

    @classmethod
    def load_from_checkpoint(cls, model_dir, device='cpu'):
        config = torch.load(f"{model_dir}/config.pt")

        # Create an instance of FantasyModel
        instance = cls(config, device)

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
