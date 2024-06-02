import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
logger = logging.getLogger(__name__)

from src.utils.utils import load_datasets, get_root_dir
from src.model.data_preprocess.football_datasets import FootballDataset

from sagemaker.s3 import S3Uploader
import boto3
import sagemaker


def prepare_data(cfg):

    # load datasets
    filenames = cfg.datasets
    logger.info(f'Preparing {len(filenames)} datasets')
    dfs = load_datasets(filenames)
    football_datasets_dict = {k: FootballDataset(k, dfs) for k in dfs.keys()}

    # get text dfs
    dfs2text_names = ['games', 'players', 'player_valuations']   # set from config
    text_dfs = {name: football_datasets_dict[name].create_text_df() for name in dfs2text_names}

    return text_dfs


def reformat_players(dfs, out_dir):
    # the json format should be
    # {"dialog":[{"content": "who is <player_name>?", "role": "user"}, {"content": <answer>, "role": "assistant"}]}...
    dialogs = []
    for i, row in dfs['players-text'].iterrows():
        text = row['text']
        pairs = text.split(', ')
        d = {}
        for p in pairs:
            try:
                k, v = p.split(': ', 1)
                d[k] = v
            except ValueError:
                continue

        curr_dialog = [{"content": f'who is {d["name"]}?', "role": 'user'}]

        # add assistant answer
        if float(d['last season']) < 2023:
            ans = (f'{d["name"]} is a former football player who played for the national team of '
                   f'{d["country of citizenship"]}. Born in {d["country of birth"]} on {d["date of birth"]}. '
                   f'He played for {d["current club name"]} as a {d["sub position"]} before he retired in '
                   f'{d["last season"]}. His highest market value was {d["highest market value in euro"]}€')
        else:
            ans = (f'{d["name"]} is a football player who plays for the national team of '
                   f'{d["country of citizenship"]}. Born in {d["country of birth"]} on {d["date of birth"]}. '
                   f'Currently he play for {d["current club name"]} as a {d["sub position"]}. '
                   f'His market value currently stand on {d["highest market value in euro"]}€.')
        curr_dialog.append({"content": ans, "role": 'assistant'})
        dialogs.append({"dialog": curr_dialog})

    with open(f'{out_dir}/players.jsonl', 'w') as f:
        for entry in dialogs:
            jl = json.dumps(entry)
            f.write(jl + '\n')


def reformat_games(dfs, out_dir):
    # "competition id: bundesliga, competition type: domestic_league, season: 2013, round: 2. Matchday,
    # date: 2013-08-18, home club name: Borussia Dortmund, away club name: Eintracht Braunschweig,
    # home club goals: 2, away club goals: 1, home club position: 1, away club position: 15,
    # home club manager name: Jürgen Klopp, away club manager name: Torsten Lieberknecht,
    # stadium: SIGNAL IDUNA PARK, attendance: 80200, referee: Peter Sippel, home club formation: 4-2-3-1,
    # away club formation: 4-3-2-1"

    # the json format should be
    # {"dialog":[{"content": "who is <player_name>?", "role": "user"}, {"content": <answer>, "role": "assistant"}]}...
    dialogs = []
    for i, row in dfs['games-text'].iterrows():
        text = row['text']
        pairs = text.split(', ')
        d = {}
        for p in pairs:
            try:
                k, v = p.split(': ', 1)
                d[k] = v
            except ValueError:
                continue

        content = f'tell me about the match between {d["home club name"]} and {d["away club name"]} on {d["date"]}'
        curr_dialog = [{"content": content, "role": 'user'}]

        # add assistant answer
        if d["competition type"] == "domestic_league":
            ans = (f'The match {d["home club name"]} vs {d["away club name"]} on {d["date"]} was a '
                   f'{d["competition id"].capitalize()}, {d["competition type"]}, game taken place on '
                   f'{d["home club name"]} stadium, the {d["stadium"]}. The match final score was '
                   f'{d["home club goals"]}:{d["away club goals"]}. '
                   f'{d["home club name"]}, led by manager {d["home club manager name"]}, played in a '
                   f'{d["home club formation"]} formation and was in position {d["home club position"]} '
                   f'on the league table. {d["away club name"]}, led by manager {d["away club manager name"]}, '
                   f'played in a {d["away club formation"]} formation and was in position {d["away club position"]} '
                   f'on the league table. There was {d["attendance"]} people in the stadium and the referee was '
                   f'{d["referee"]}.')
        else:
            ans = (f'The match {d["home club name"]} vs {d["away club name"]} on {d["date"]} was a '
                   f'{d["competition id"].capitalize()}, {d["competition type"]}, game taken place on '
                   f'{d["home club name"]} stadium, the {d["stadium"]}. The match final score was '
                   f'{d["home club goals"]}:{d["away club goals"]}. '
                   f'{d["home club name"]}, led by manager {d["home club manager name"]}, played in a '
                   f'{d["home club formation"]} formation. {d["away club name"]}, led by manager '
                   f'{d["away club manager name"]}, played in a {d["away club formation"]} formation. '
                   f'There was {d["attendance"]} people in the stadium and the referee was {d["referee"]}.')

        curr_dialog.append({"content": ans, "role": 'assistant'})
        dialogs.append({"dialog": curr_dialog})

    with open(f'{out_dir}/games.jsonl', 'w') as f:
        for entry in dialogs:
            jl = json.dumps(entry)
            f.write(jl + '\n')


def convert_to_json_format():

    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data')
    out_dir = f'{data_dir}/jsons'
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    # filenames = ['players-text', 'games-text']
    # dfs = {fn: pd.read_csv(f'{data_dir}/{fn}.csv', keep_default_na=False) for fn in filenames
    #        if Path(f'{data_dir}/{fn}.csv').exists()}

    # reformat_players(dfs, out_dir)
    # reformat_games(dfs, out_dir)
    # upload data to s3
    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    bucket = 'deep-learning-projects/sagemaker/FootballGPT_Data/jsons'
    local_data_file = f'{out_dir}/games.jsonl'
    pref = 'sagemaker/FootballGPT_Data/jsons'
    datapath = sess.upload_data(path="")
    # train_data_location = f"s3://{output_bucket}/oasst_top1"
    # S3Uploader.upload(local_data_file, train_data_location)
    # print(f"Training data: {train_data_location}")

if __name__ == '__main__':
    convert_to_json_format()
