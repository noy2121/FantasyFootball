from typing import Dict
import pandas as pd
from wrangler_utils import get_club_name_by_club_id
from datasets_structure import events_cols


def split_cards_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split 'Cards' event type to 'yellow-card', 'red-card', 'second-yellow-card' event types.
    """
    new_events_df = events_df.copy()

    cards_mask = new_events_df['type'] == 'Cards'
    cards_df = new_events_df[cards_mask]

    yellows = cards_df[cards_df['description'].str.contains('yellow', case=False, na=False) &
                       ~cards_df['description'].str.contains('second yellow', case=False, na=False)].copy()
    second_yellows = cards_df[cards_df['description'].str.contains('second yellow', case=False, na=False)].copy()
    reds = cards_df[cards_df['description'].str.contains('red', case=False, na=False)].copy()

    assert (yellows['type'] == 'Cards').all(), f'Some yellow card events have wrong event type !'
    assert (second_yellows['type'] == 'Cards').all(), f'Some second yellow card events have wrong event type !'
    assert (reds['type'] == 'Cards').all(), f'Some red card events have wrong event type !'

    # change event type Cards -> yellow/second/red
    yellows['type'] = 'yellow-card'
    second_yellows['type'] = 'second-yellow-card'
    reds['type'] = 'red-card'

    new_cards = pd.concat([yellows, second_yellows, reds])
    new_events_df = pd.concat([new_events_df[~cards_mask], new_cards]).reset_index(drop=True)
    new_events_df.rename(columns={'game_event_id': 'event_id', 'type': 'event_type'}, inplace=True)

    return new_events_df


def create_events_df(raw_events_df: pd.DataFrame, games_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    print('Extract Events data...')
    # filter by game_id
    # no need to filter by year or club_id, games_df has already been filtered this way
    relevant_game_ids = games_df['game_id'].tolist()
    events_df = raw_events_df[raw_events_df['game_id'].isin(relevant_game_ids)]

    events_df = events_df.assign(date=pd.to_datetime((events_df['date'])))
    cutoff_date = pd.Timestamp(f'{start_year}-08-01')
    assert (events_df['date'] >= cutoff_date).all(), f'Some dates in the DataFrame are before {cutoff_date}'

    events_df = split_cards_events(events_df)
    events_df = events_df[events_cols]

    return events_df


def format_line(col, val, dfs):
    def safe_get(df, cond, column):
        try:
            res = df.loc[cond, column].iloc[0]
            return 'Unknown' if pd.isna(res) else res
        except (IndexError, KeyError):
            return 'Unknown'

    def handle_game_id():
        home_id = safe_get(dfs['games'], dfs['games']['game_id'] == val, 'home_club_id')
        away_id = safe_get(dfs['games'], dfs['games']['game_id'] == val, 'away_club_id')
        home_name = get_club_name_by_club_id(home_id)
        away_name = get_club_name_by_club_id(away_id)
        return "match", f"{home_name} vs {away_name}"

    def handle_club_id():
        return "club_name", get_club_name_by_club_id(val)

    def handle_player_id():

        return "player_name", safe_get(dfs['players'], dfs['players']['player_id'] == val, 'player_name')

    def handle_player_in_id():
        if pd.isna(val):
            return None, None
        return "player_in_name", safe_get(dfs['players'], dfs['players']['player_id'] == val, 'player_name')

    def handle_player_assist_id():
        if pd.isna(val):
            return None, None
        return "assist_player_name", safe_get(dfs['players'], dfs['players']['player_id'] == val, 'player_name')

    def handle_date():
        return col, val.date()

    handlers = {
        'date': handle_date,
        'game_id': handle_game_id,
        'club_id': handle_club_id,
        'player_id': handle_player_id,
        'player_in_id': handle_player_in_id,
        'player_assist_id': handle_player_assist_id
    }

    handler = handlers.get(col)
    if handler:
        new_col, new_val = handler()
        if new_col is None and new_val is None:
            return ''
        return f"{new_col}: {new_val}"
    return f"{col}: {val}"


def create_text_events_df(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    print('Convert Events data to text...')

    def format_row(row):
        return ', '.join(f"{format_line(col, val, dfs)}" for col, val in row.items() if col != 'event_id')

    return pd.DataFrame({'text': dfs['events'].apply(format_row, axis=1)})
