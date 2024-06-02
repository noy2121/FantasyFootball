import os
import torch
import numpy as np
import pandas as pd
import hydra
import json
import s3fs

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from pathlib import Path
from peft import LoraConfig, PrefixTuningConfig, AdaLoraConfig, LoKrConfig, get_peft_model

from src.utils.utils import set_random_seed, get_root_dir
from src.model.data_preprocess.football_torch_dataset import FootballTorchDataset
from src.model.data_preprocess.data_preperation import prepare_data


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
    Path(out_dir).mkdir(parents=True, exists_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optim,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
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


def tokenize(x, tokenizer):
    pass


@hydra.main(config_path='../config', config_name='conf')
def train(cfg):

    root_dir = get_root_dir()

    # check available gpu
    # check_gpu()

    # set up quantization config
    device_map = {"": 0}
    # bnb_config = config_quantization(cfg.model)

    # load base model
    base_model_name = cfg.model.model_name
    print(f'load model {base_model_name}')
    foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name, use_auth_token=cfg.model.huggingface_token)
    # foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name,
    #                                                         quantization_config=bnb_config,
    #                                                         device_map=device_map,
    #                                                         use_auth_token=cfg.model.huggingface_token)
    foundation_model.config.use_cache = False
    foundation_model.config.pretraining_tp = 1

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    # load data
    data_location = f'{cfg.data.json_files_path}'
    players_ds = load_dataset('json', data_files=f'{data_location}/players.jsonl', split='train')
    games_ds = load_dataset('json', data_files=f'{data_location}/games.jsonl', split='train')
    train_ds = concatenate_datasets([players_ds, games_ds])
    # tokenized_ds = train_ds.map(lambda s: tokenize(s, tokenizer), batch=True)

    # define peft methods and configurations
    print(f'configure {cfg.peft.method_name} for PEFT')
    peft_config = set_peft_config(cfg.peft)
    peft_model = get_peft_model(foundation_model, peft_config)
    print(peft_model.print_trainable_parameters())

    # define training args
    training_args = set_training_config(cfg.train, root_dir)

    # train
    print('start training')
    print("=" * 80)
    trainer = SFTTrainer(
        model=foundation_model,
        train_dataset=train_ds,
        peft_config=peft_config,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )
    trainer.train()
    print("=" * 80)

    print('save fine-tuned model')
    trainer.model.save_pretrained(f'{base_model_name}-football-{cfg.peft.method_name}-ft')


if __name__ == '__main__':
    set_random_seed()
    train()
