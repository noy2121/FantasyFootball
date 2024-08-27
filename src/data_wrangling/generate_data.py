import numpy as np
import random
from datetime import datetime

from system_prompts import round_dates


def get_weighted_teams(clubs_df, season, num_teams):

    # Filter clubs that participated in Champions League this season
    cl_clubs = clubs_df[f'{season}_champions_league_place'].to_numpy()
    club_names = clubs_df['club_name'].to_numpy()

    # Create weights based on their performance
    weight_map = {
        'Winners': 25, 'Final': 20, 'Semi-Final': 20, 'Quarter-Final': 15,
        'Round of 16': 10, 'Group Stage': 5, 'Not Qualified': 0.1
    }
    weights = np.array([weight_map.get(place, 1) for place in cl_clubs])

    # Normalize weights
    weights = weights / weights.sum()

    # Sample clubs based on weights
    selected_indices = np.random.choice(
        len(cl_clubs),
        size=num_teams,
        replace=False,
        p=weights
    )
    selected_clubs = club_names[selected_indices]

    return selected_clubs


def create_sample(clubs_df, num_matches, season, round_name, date):
    teams = get_weighted_teams(clubs_df, season, num_matches * 2)
    matches = [f"{teams[i]} vs {teams[i + 1]}" for i in range(0, len(teams), 2)]

    sample = (
        f"matches: [{', '.join(matches)}]\n"
        f"round: {round_name}\n"
        f"season: {season}\n"
        f"date: {date}"
    )
    return sample


def is_valid_sample(sample):
    # Add any additional validation checks here
    return len(sample.split('\n')) == 4 and all(key in sample for key in ['matches:', 'round:', 'season:', 'date:'])


def generate_samples(clubs_df, samples_per_season=1000):

    # Generate samples
    train_samples = []
    test_samples = []
    seasons = [f"{year}/{str(year + 1)[-2:]}" for year in range(2017, 2024)]
    matches_in_round = {
        "Group Stage": 16,
        "Round of 16": 8,
        "Quarter-Final": 4,
        "Semi-Final": 2,
        "Final": 1
    }

    for season in seasons:
        for _ in range(samples_per_season):
            round_name = np.random.choice(list(round_dates.keys()), p=[0.25, 0.25, 0.2, 0.2, 0.1])
            date = np.random.choice(round_dates[round_name])
            num_matches = matches_in_round[round_name]
            if round_name == 'Group Stage':
                date = datetime.strptime(f'{date}/{season[2:4]}', '%d/%m/%y').strftime('%Y-%m-%d')
            else:
                date = datetime.strptime(f'{date}/{season[-2:]}', '%d/%m/%y').strftime('%Y-%m-%d')
            sample = create_sample(clubs_df, num_matches, season, round_name, date)

            if is_valid_sample(sample):
                if season == '2023/24':
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

    train_dict, test_dict = {}, {}
    for i in range(len(train_samples)):
        train_dict[f'{i:04}'] = train_samples[i]

    for j in range(len(test_samples)):
        test_dict[f'{i+j+1:04}'] = test_samples[j]

    return train_dict, test_dict

