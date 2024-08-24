

matches_prompt = [
    ("Copenhagen", "Manchester City"),
    ("RB Leipzig", "Real Madrid"),
    ("Paris Saint-Germain", "Real Sociedad"),
    ("Lazio", "Bayern Munich"),
    ("PSV Eindhoven", "Borussia Dortmund"),
    ("Inter Milan", "Atletico Madrid"),
    ("Porto", "Arsenal"),
    ("Napoli", "Barcelona")
]
sample_user_prompt = (
    "matches: [Copenhagen vs Manchester City, "
    "RB Leipzig vs Real Madrid, "
    "Paris Saint-Germain vs Real Sociedad, "
    "Lazio vs Bayern Munich, "
    "PSV Eindhoven vs Borussia Dortmund, "
    "Inter Milan vs Atletico Madrid, "
    "Porto vs Arsenal, "
    "Napoli vs Barcelona]\n"
    "round: Group Stage\n"
    "season: 2022/23\n"
    "date: 2023-09-17")

instruction_prompt = ("You are a participant in a fantasy football league. "
                      "Your task is to build a team consist of 11 players to finish as high as possible. "
                      "You must follow the rules of the game, and cannot break them.")

rules_prompt = ("Squad: The team must include 11 selected players: "
                "1 Goalkeeper, "
                "3-5 defenders, "
                "3-5 midfielders, "
                "1-3 attackers. "
                "The number of players that can be selected from each team is: 3. "
                "Budget: 125 million. "
                "Points: Each player earns points based on the actual performance on the field. "
                "A player who played less than 60 minutes: 1 point. "
                "A player who played at least 60 minutes (extra time does not count. Overtime, if it occurs, does count): 2 points. "
                "Goal scored by a goalkeeper: 7 points. "
                "Goal scored by a defender: 6 points. "
                "Goal scored by a midfielder: 5 points. "
                "Goal scored by an attacker: 4 points. "
                "Bonus for scoring goals: If a player score more than 1 goal he will recieve bonus points in this manner: "
                "2 goals = +1 bonus point. Hat-trick = +2 bonus points. Four goals = +3 bonus points, and so on. "
                "Assist: 3 points. "
                "Goalkeeper who kept a clean sheet, and played at least 60 minutes, excluding added time, including overtime: 4 points. "
                "Defender who kept a clean sheet (assuming played at least 60 minutes, excluding added time, including overtime): 4 points. "
                "Player who caused a penalty: -4 points. "
                "Player who won a penalty: 2 points. "
                "Goalkeeper who saved a penalty: 4 points. "
                "Player who missed a penalty: -4 points. "
                "Player who scored an own goal: -5 points. "
                "Yellow card: -1 point. "
                "Two yellow cards leading to a red card: -3 points. "
                "Red card: -3 points. "
                "Negative bonus for conceding goals: conceding one goal cancel the clean sheet points. from the second goal onwards, -1 point for each goal conceded. "
                "Overtime is counted in the scoring; penalty shootouts after overtime are not counted. ")

one_shot_example = ("Goalkeeper: Andriy Lunin\n"
                    "Defence: Matteo Darmian, Federico Dimarco, Joao Mario \n"
                    "Midfield: Jamal Musiala, Dani Olmo, Marcel Sabitzer \n"
                    "Attack: Bukayo Saka, Erling Haaland, Lamine Yamal \n"
                    "Budget used: 125M/125M"
                    "season: 2023/24")

player_entry_format = ("Player: {}\n"
                       "Position: {}\n"
                       "Club: {}\n"
                       "Last 5 games performance:\n"
                       "\tGoals: {}\n"
                       "\tAssists: {}\n"
                       "\tLineups: {}\n"
                       "Seasonal performance:\n"
                       "\tGoals: {}\n"
                       "\tAssists: {}\n"
                       "\tLineups: {}\n")

club_entry_format = ("Club: {}\n"
                     "Last 5 Champions League games performance:{}\n"
                     "Last 5 Domestic League games performance:{}\n"
                     "Seasonal performance:\n"
                     "\tWins: {}\n"
                     "\tLoses: {}\n"
                     "\tTies: {}\n"
                     "Champions League titles: {}\n")

round_dates = {'Group Stage': ['06/09', '13/09', '04/10', '11/10', '25/10', '01/11'],
               'Round of 16': ['14/02', '07/03'],
               'Quarter-Final': ['11/04', '18/04'],
               'Semi-Final': ['09/05', '16/05'],
               'Final': ['10/06']}
