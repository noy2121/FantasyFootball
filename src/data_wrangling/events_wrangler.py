import pandas as pd

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
    print('Extract events data...')
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



