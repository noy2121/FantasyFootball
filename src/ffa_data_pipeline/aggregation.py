import numpy as np


def aggregate_data(dfs):

    players_df, clubs_df, player_prices_df = dfs["players"], dfs["clubs"], dfs["player_prices"]

    # Select and rename relevant columns
    players_df = players_df[["player_id", "name", "current_club_id", "position"]]
    players_df.rename(columns={
        "name": "player_name",
        "current_club_id": "club_id"
    }, inplace=True)

    clubs_df = clubs_df[["club_id", "name"]]
    clubs_df.rename(columns={
        "name": "club_name"
    }, inplace=True)

    player_prices_df = player_prices_df[["Player", "Price"]]
    player_prices_df.rename(columns={
        "Player": "player_name",
        "Price": "price"
    }, inplace=True)

    # Merge datasets
    merged_df = players_df.merge(clubs_df, on="club_id", how="inner")

    # Add "price" column with values from a normal distribution
    mean_price = 8
    std_dev = 2
    min_price = 3
    max_price = 15
    merged_df['price'] = np.clip(np.random.normal(mean_price, std_dev, size=len(merged_df)),
                                 min_price, max_price).astype(int)

    # Reorder columns
    processed_df = merged_df[["player_id", "player_name", "club_id", "club_name", "position", "price"]]

    return processed_df
