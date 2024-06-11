import os
import torch
import hydra
import transformers
import plotly.graph_objects as go
import evaluate

from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers import BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling

from src.finetune.trainer import check_gpu, config_quantization, tokenize
from src.finetune.evaluator import compute_metrics
from src.utils.utils import set_random_seed, get_root_dir


def load_ft_models(models_dir, base_model_name, bnb_config, device_map, cache_dir, auth_token):
    models_dict, tokenizers_dict = {}, {}

    # load base model
    print(f'load pre-trained model {base_model_name}')
    foundation_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                            quantization_config=bnb_config,
                                                            device_map=device_map,
                                                            cache_dir=cache_dir,
                                                            token=auth_token)
    foundation_model.config.pretraining_tp = 1
    models_dict[base_model_name] = foundation_model

    # load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizers_dict[base_model_name] = tokenizer

    for model_name in os.listdir(models_dir):
        # load using peft
        model = PeftModel.from_pretrained(foundation_model, f'{models_dir}/model_name')
        model = model.merge_and_unload()
        models_dict[model_name] = model

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        tokenizers_dict[model_name] = tokenizer

    return models_dict, tokenizers_dict


def generate_sample(model, tokenizer, model_name):
    input = tokenizer("Cristiano Ronaldo is a ", return_tensors="pt")

    outputs = model.generate(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        max_new_tokens=30,
        eos_token_id=tokenizer.eos_token_id
    )

    print(f'generate sample from model: {model_name}')
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print('\n')


def evaluate_model(model, tokenized_val_ds, tokenizer, out_dir):
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        logging_steps=500,
        save_steps=500,
        group_by_length=True
    )

    evaluator = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )
    evaluator.evaluate()


@hydra.main(config_path='../config', config_name='conf')
def compare(cfg):
    root_dir = get_root_dir()

    # check available gpu
    check_gpu()

    # set up quantization config
    device_map = {"": 0}
    bnb_config = config_quantization(cfg.model)

    base_model_name = cfg.model.model_name
    cache_dir = os.path.join(root_dir, cfg.model.cache_dir)
    with open(os.path.join(root_dir, cfg.model.huggingface_token_filepath)) as f:
        auth_token = f.readline().rstrip()

    # load models
    models_dir = os.path.join(root_dir, cfg.evaluate.models_dir)
    models_dict, tokenizers_dict = load_ft_models(models_dir, base_model_name, bnb_config, device_map, cache_dir,
                                                  auth_token)

    # generate samples for every model
    for model_name, model in models_dict.items():
        generate_sample(model, tokenizers_dict[model_name], model_name)

    # load datasets
    data_location = os.path.join(root_dir, cfg.data.data_files_path)
    if 'bloomz' in base_model_name:
        data_files = {'validation': [f'{data_location}/val_games.csv', f'{data_location}/val_players.csv']}
        ds = load_dataset('csv', data_files=data_files)
        val_ds = ds["validation"]
        print(ds)
    else:
        raise ValueError(f'model name {base_model_name} not excepted! please choose between [Mistral-7B, bloomz-560m]')

    # evaluate each model
    for model_name, model in models_dict.items():

        # tokenize data
        print('\ntokenize dataset')
        tokenizer = tokenizers_dict[model_name]
        tokenized_val_ds = val_ds.map(lambda s: tokenize(s, tokenizer), batched=True)
        tokenized_val_ds = tokenized_val_ds.remove_columns(['text'])
        tokenized_val_ds.set_format('torch')

        # perform evaluation on the validation set
        out_dir = f'{cfg.evaluate.out_dir}/{model_name}'
        out_dir = os.path.join(root_dir, out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_dir = str(out_dir)

        evaluate_model(model, tokenized_val_ds, tokenizer, out_dir)


    # TODO: plot perplexity & rouge scores


if __name__ == '__main__':
    set_random_seed()
    compare()
