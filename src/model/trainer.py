import os
import logging

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
streamhandler = logging.StreamHandler()
streamhandler.setLevel(logging.DEBUG)
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

import numpy as np
import pandas as pd
import hydra

from sagemaker import hyperparameters
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from pathlib import Path
from peft import LoraConfig, PrefixTuningConfig, AdaLoraConfig, LoKrConfig

from src.utils.utils import set_random_seed, get_root_dir
from src.model.data_preprocess.football_torch_dataset import FootballTorchDataset


def config_quantization(cfg):
    logger.info('configure quantization parameters')
    conf = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_compute_dtype=cfg.quant_compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
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


def tokenize(cfg, tokenizer):

    # load data from s3 bucket
    data_location = f's3://{cfg.bucket}/{cfg.folder}'
    if cfg.datasets is not None:  # read only specified datasets
        dfs = [pd.read_csv(f'{data_location}/{ds}.csv') for ds in cfg.datasets]
    else:  # read all available datasets
        dfs = [pd.read_csv(f'{data_location}/{ds}') for ds in os.listdir(data_location)
               if Path(f'{data_location}/{ds}.csv').exists() and Path(f'{data_location}/{ds}').suffix == '.csv']

    if len(dfs) > 1:
        combined_df = pd.concat([df for df in dfs], ignore_index=True)
    combined_df = combined_df.dropna()  # Drop rows with missing values

    # tokenize all data
    inputs = tokenizer(combined_df['text'].tolist(), max_length=512, truncation=True, padding='max_length',
                       return_tensors='pt')

    # transform datasets to pytorch Dataset instance
    ds = FootballTorchDataset(inputs)

    return ds


@hydra.main(config_path='../config', config_name='conf')
def train(cfg):

    root_dir = get_root_dir()

    # set up quantization config
    device_map = {"": 0}
    bnb_config = config_quantization(cfg.model)

    # load base model
    base_model_name = cfg.model.model_name
    logger.info(f'load model {base_model_name}')
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
    train_ds = tokenize(cfg.data, tokenizer)

    # define peft methods and configurations
    logger.info(f'configure {cfg.peft.method_name} for PEFT')
    peft_config = set_peft_config(cfg.peft)

    # define training args
    training_args = set_training_config(cfg.train, root_dir)

    # train
    logger.info('start training')
    print("=" * 80)
    trainer = SFTTrainer(
        model=foundation_model,
        train_dataset=train_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )
    trainer.train()
    print("=" * 80)

    logger.info('save fine-tuned model')
    trainer.model.save_pretrained(f'{base_model_name}-football-{cfg.peft.method_name}-ft')


if __name__ == '__main__':
    set_random_seed()
    train()
