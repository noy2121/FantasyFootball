import os
import torch
import logging
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import peft
import numpy as np
import pandas as pd

from pathlib import Path
from peft import LoraConfig, PrefixTuningConfig, AdaLoraConfig, LoKrConfig
from src.utils.utils import set_random_seed, get_root_dir
from src.model.data_preprocess.football_torch_dataset import FootballTorchDataset
from src.model.data_preprocess.data_preperation import prepare_data


def config_quantization():
    conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=False
    )

    # # check GPU compatibility with bfloat16
    # compute_dtype = getattr(torch, 'float16')
    # if compute_dtype == torch.float16:
    #     major, _ = torch.cuda.get_device_capability()
    #     if major >= 8:
    #         print("=" * 80)
    #         print("Your GPU supports bfloat16: accelerate training with bf16=True")
    #         print("=" * 80)

    return conf


def set_peft_config(method_name):
    # define peft methods and configurations
    if method_name == 'lora':
        config = LoraConfig(r=64, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.1, bias='none')
    elif method_name == 'adalora':
        config = AdaLoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.01)
    elif method_name == 'lokr':
        config = LoKrConfig(r=8, lora_alpha=32, target_modules=["c_attn", "c_proj", "c_fc"])
    else:
        config = PrefixTuningConfig(num_virtual_tokens=20, token_dim=768, num_transformer_submodules=1,
                                    num_attention_heads=12, num_layers=12, encoder_hidden_size=768)

    return config


def set_training_config(root_dir):
    checkpoint_name = './results'
    out_dir = os.path.join(root_dir, checkpoint_name)
    Path(out_dir).mkdir(parents=True, exists_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim='paged_adamw_32bit',
        save_steps=1000,
        logging_steps=1000,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type='constant',
        report_to="tensorboard"
    )

    return training_args


def train():

    root_dir = get_root_dir()
    # set up quantization config
    device_map = {"": 0}
    bnb_config = config_quantization()

    # load base model
    base_model_name = 'bigscience/bloomz-560m'  # bigscience/bloomz-560m
    foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                            quantization_config=bnb_config,
                                                            device_map=device_map)
    foundation_model.config.use_cache = False
    foundation_model.config.pretraining_tp = 1

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # load data, tokenize and prepare it for training
    text_dfs = prepare_data()
    combined_text_df = pd.concat([df for df in text_dfs.values()], ignore_index=True)
    inputs = tokenizer(combined_text_df['text'].tolist(), max_length=512, truncation=True, padding='max_length',
                       return_tensors='pt')    # transform datasets to pytorch Dataset instance
    football_ds = FootballTorchDataset(inputs)

    # define peft methods and configurations
    method_name = 'lora'  # load from config
    peft_config = set_peft_config(method_name)

    # define training args
    training_args = set_training_config(root_dir)

    # train
    trainer = SFTTrainer(
        model=foundation_model,
        train_dataset=football_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )
    trainer.train()

    trainer.model.save_pretrained(f'{base_model_name}-football-{method_name}-ft')


if __name__ == '__main__':
    set_random_seed()
    train()
