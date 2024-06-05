import os
import sys

import torch
import numpy as np
import pandas as pd
import hydra
import json
import s3fs

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers import BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from pathlib import Path
from peft import LoraConfig, PrefixTuningConfig, AdaLoraConfig, LoKrConfig, get_peft_model

from src.finetune.evaluator import compute_metrics
from src.utils.utils import set_random_seed, get_root_dir


def check_gpu():
    is_avialabe = torch.cuda.is_available()
    if not is_avialabe:
        raise 'No GPU has found! Exit!'
    device_count = torch.cuda.device_count()
    curr_device = torch.cuda.current_device()
    print(f'Device count: {device_count}')
    print(f'Current device: {curr_device}')
    print(f'Device name: {torch.cuda.get_device_name(curr_device)}')


def config_quantization(cfg):
    print('configure quantization parameters')
    conf = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_compute_dtype=cfg.quant_compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )

    return conf


def set_peft_config(cfg):
    method_name = cfg.method

    # define peft methods and configurations
    if method_name == 'lora':
        config = LoraConfig(r=cfg.r, lora_alpha=cfg.lora_a, target_modules=cfg.target_modules,
                            lora_dropout=cfg.dropout, bias='none')
    elif method_name == 'adalora':
        config = AdaLoraConfig(r=cfg.r, lora_alpha=cfg.lora_a, target_modules=cfg.target_modules,
                               lora_dropout=cfg.dropout)
    elif method_name == 'lokr':
        config = LoKrConfig(r=cfg.r, lora_alpha=cfg.lora_a, target_modules=cfg.target_modules)
    else:
        config = PrefixTuningConfig(num_virtual_tokens=20, token_dim=768, num_transformer_submodules=1,
                                    num_attention_heads=12, num_layers=12, encoder_hidden_size=768)

    return config


def set_training_config(cfg, root_dir):
    checkpoint_name = cfg.checkpoint_name
    out_dir = os.path.join(root_dir, checkpoint_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        no_cuda=True,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy='epoch'
    )

    return training_args


def tokenize(x, tokenizer):
    return tokenizer(x['text'], padding='max_length', truncation=True)


@hydra.main(config_path='../config', config_name='conf')
def train(cfg):
    root_dir = get_root_dir()

    # check available gpu
    # check_gpu()

    # set up quantization config
    device_map = {"": 0}
    # bnb_config = config_quantization(cfg.finetune)

    with open(os.path.join(root_dir, cfg.model.huggingface_token_filepath)) as f:
        auth_token = f.readline().rstrip()

    # load base finetune
    base_model_name = cfg.model.model_name
    print(f'load pre-trained model {base_model_name}')
    foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=cfg.model.cache_dir,
                                                            token=auth_token)
    # foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name,
    #                                                         quantization_config=bnb_config,
    #                                                         device_map=device_map,
    #                                                         use_auth_token=cfg.finetune.huggingface_token.txt)
    foundation_model.config.pretraining_tp = 1

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cfg.model.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    # prepare data - data format for bloomz model is different from Mistral-7B
    data_location = f'{cfg.data.data_files_path}'
    if 'bloomz' in base_model_name:
        data_files = {'train': [f'{data_location}/train_games.csv', f'{data_location}/train_players.csv'],
                      'validation': [f'{data_location}/val_games.csv', f'{data_location}/val_players.csv']}
        ds = load_dataset('csv',
                                data_files=data_files)
        train_ds = ds["train"]
        val_ds = ds["validation"]
        print(ds)
        sys.exit(2)

    elif 'Mistral' in base_model_name:
        players_ds = load_dataset('json',
                                  data_files=[f'{data_location}/jsons/players.jsonl',
                                              f'{data_location}/jsons/games.jsonl'],
                                  split='train')
    else:
        raise ValueError(f'model name {base_model_name} not excepted! please choose between [Mistral-7B, bloomz-560m]')

    # tokenized data
    tokenized_train_ds = train_ds.map(lambda s: tokenize(s, tokenizer), batched=True)
    tokenized_train_ds = tokenized_train_ds.remove_columns(['text'])
    tokenized_train_ds.set_format('torch')

    tokenized_val_ds = val_ds.map(lambda s: tokenize(s, tokenizer), batched=True)
    tokenized_val_ds = tokenized_val_ds.remove_columns(['text'])
    tokenized_val_ds.set_format('torch')

    # define peft methods and configurations
    print(f'configure {cfg.peft.method} for PEFT')
    peft_config = set_peft_config(cfg.peft)
    peft_model = get_peft_model(foundation_model, peft_config)
    print(peft_model.print_trainable_parameters())

    # define training args
    training_args = set_training_config(cfg.train, root_dir)

    # train
    print('start training')
    print("=" * 80)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics
    )
    trainer.train()
    print("=" * 80)

    print('save fine-tuned finetune')
    trainer.model.save_pretrained(f'{base_model_name}-football-{cfg.peft.method_name}-ft')


if __name__ == '__main__':
    set_random_seed()
    train()
