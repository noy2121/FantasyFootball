
players_cols = [
    "player_id",
    "player_name",
    "club_id",
    "position",
    "date_of_birth",
    "goals_in_last_three_seasons",
    "total_goals",
    "assists_in_last_three_seasons",
    "total_assists",
    "yellow_cards_in_last_three_seasons",
    "total_yellow_cards",
    "red_cards_in_last_three_seasons",
    "total_red_cards",
    "clean_sheets_in_last_three_seasons",
    "total_clean_sheets",
    "starting_lineups",
    "substitute",
    "price"
]

clubs_cols = [
    "club_id",
    "club_name",
    "champions_league_rank",
    "prev_season_domestic_league_place",
    "prev_season_champions_league_place",
    "num_champions_league_titles"
]

games_cols = [
    "game_id",
    "competition_type",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_goals",
    "away_club_goals",
    "home_club_formation",
    "away_club_formation",
    "aggregate"
]

events_cols = [
    "event_id",
    "event_type",
    "game_id",
    "club_id",
    "date",
    "minute",
    "player_id",
    "player_in_id",
    "player_assist_id"
]

teams = {
    "Manchester City",
    "Liverpool",
    "Manchester United",
    "Aston Villa",
    "Arsenal",
    "Bologna FC 1909",
    "Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Bayer Leverkusen",
    "Paris Saint-Germain",
    "AS Monaco",
    "Stade Brestois 29",
    "Lille",
    "Juventus",
    "AC Milan",
    "Inter Milan",
    "Napoli",
    "Atalanta",
    "Girona FC",
    "Real Madrid",
    "Barcelona",
    "Atletico Madrid",
    "Sporting CP",
    "Benfica",
    "Feyenoord Rotterdam",
    "PSV Eindhoven",
    "Club Brugge",
    "VfB Stuttgart",
    "Shakhtar Donetsk",
    "Dynamo Kyiv",
    "Celtic"
}

text_players_cols = ["player_name",
                     "club_name",
                     "position",
                     "date_of_birth",
                     "goals_in_last_three_seasons",
                     "total_goals",
                     "assists_in_last_three_seasons",
                     "total_assists",
                     "yellow_cards_in_last_three_seasons",
                     "total_yellow_cards",
                     "red_cards_in_last_three_seasons",
                     "total_red_cards",
                     "clean_sheets_in_last_three_seasons",
                     "total_clean_sheets",
                     "lineups",
                     "price"
                     ]

prev_champions_league_vals = ["Winners", "Second", "Semi-Finals", "Quarter-Finals", "Last-16", "Group-Stage", "Not Qualified"]

competition_vals = ["domestic_league", "champions_league", "other"]
