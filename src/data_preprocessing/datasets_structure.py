
players = [
    "player_id",
    "player_name",
    "position",
    "club_id",
    "goals_per_year",
    "total_goals",
    "assists_per_year",
    "total_assists",
    "yellow_cards_per_year",
    "total_yellow_cards",
    "red_cards_per_year",
    "total_red_cards",
    "xg_per_year",
    "total_xg",
    "appearances",
    "price"
]

clubs = [
    "club_id",
    "club_name",
    "lineup",
    "domestic_league_place",
    "champions_league_place",
    "rating",
    "champions_league_title"
]

matches = [
    "match_id",
    "competition",
    "date",
    "home_team",
    "away_team",
    "score"
]

events = [
    "event_id",
    "event_type",  # goal, assist, yellow-card, red-card, substitute
    "match_id",
    "club_id",
    "minute",
    "player_id",   # the player that cause the event. in case of substitute it's the player_id of the player getting out
    "player_in_id",
    "player_assist_id"
]

teams = [
    "Manchester City",
    "Liverpool",
    "Chelsea",
    "Tottenham Hotspur",
    "Arsenal",
    "Manchester United",
    "Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Bayer Leverkusen",
    "Paris Saint-Germain",
    "Marseille",
    "Lyon",
    "Lille",
    "Juventus",
    "AC Milan",
    "Inter Milan",
    "Napoli",
    "Real Madrid",
    "Barcelona",
    "Atletico Madrid",
    "Sevilla",
    "Porto",
    "Benfica",
    "Ajax",
    "PSV Eindhoven",
    "Club Brugge",
    "Anderlecht",
    "Shakhtar Donetsk",
    "Dynamo Kyiv",
    "Atalanta",
    "Celtic"
]