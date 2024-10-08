

sample_user_prompt = (
    "matches: [Copenhagen vs Manchester City, "
    "RB Leipzig vs Real Madrid, "
    "Paris Saint-Germain vs Real Sociedad, "
    "Lazio vs Bayern Munich, "
    "PSV Eindhoven vs Borussia Dortmund, "
    "Inter Milan vs Atletico Madrid, "
    "Porto vs Arsenal, "
    "Napoli vs Barcelona]\n"
    "round: Round of 16\n"
    "season: 2022/23\n"
    "date: 2023-09-17")

instruction_prompt = (
    "You are a participant in a fantasy football league. "
    "Your task is to build a team consist of 11 players to finish as high as possible. "
    "You must follow the rules of the game, and cannot break them.\n"
    "The following is an example of the output format. Your output must have this format."
    "<OUTPUT STARTS>"
    "Team:\n"
    "\tGoalkeeper: <PLAYER NAME> (<PLAYER COST>)\n"
    "\tDefence: 3*<PLAYER NAME> (<PLAYER COST>)\n"
    "\tMidfield: 4*<PLAYER NAME> (<PLAYER COST>)\n"
    "\tAttack: 3*<PLAYER NAME> (<PLAYER COST>)\n"
    "Budget used: <SUM OF COSTS>M/110M"
    "<OUTPUT ENDS>"
)

short_rules_prompt = (
    "Squad: The team must include 11 selected players: "
    "1 Goalkeeper, "
    "3-5 defenders, "
    "3-5 midfielders, "
    "1-3 attackers. "
    "The number of players that can be selected per team varies in each round: "
    "group stage: 2 players "
    "round of 16: 3 players "
    "quarter-final: 4 "
    "semi-final: 6 "
    "final: 8"
    "The budget varies in each round:"
    "group stage: 100M "
    "round of 16: 110m "
    "quarter-final: 110M "
    "semi-final: 120m "
    "final: 120M "
    "Points: Each player earns points based on the actual performance on the field. "
    "You should prefer players that has a many goals or assists, and players from strong teams."
)


full_rules_prompt = (
    "Squad: The team must include 11 selected players: "
    "1 Goalkeeper, "
    "3-5 defenders, "
    "3-5 midfielders, "
    "1-3 attackers. "
    "The number of players that can be selected per team varies in each round: "
    "group stage: 2 players "
    "round of 16: 3 players "
    "quarter-final: 4 "
    "semi-final: 6 "
    "final: 8."
    "Notice that each player can only be chosen once! "
    "The budget varies in each round:"
    "group stage: 100M "
    "round of 16: 110m "
    "quarter-final: 110M "
    "semi-final: 120m "
    "final: 120M "
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
    "Overtime is counted in the scoring; penalty shootouts after overtime are not counted."
)

one_shot_example = ("Team:\n"
                    "\tGoalkeeper: Andriy Lunin (6M)\n"
                    "\tDefence: Matteo Darmian (6M), Federico Dimarco (7M), Joao Mario (6M)\n"
                    "\tMidfield: Jamal Musiala (12M), Dani Olmo(10M), Marcel Sabitzer(9M), Phil Foden (13M)\n"
                    "\tAttack: Bukayo Saka (14M), Erling Haaland (15M), Lamine Yamal (11M)\n"
                    "Budget used: 109M/125M"
                    )

player_entry_format = ("Player: {}, "
                       "Position: {}, "
                       "Club: {}, "
                       "Cost: {}, "
                       "Last 5 games average score: {}, "
                       "Seasonal average score: {}."
                       )
club_entry_format = ("Club: {}, "
                     "Last 5 Champions League games:{}, "
                     "Last 5 Domestic League games:{}, "
                     "Seasonal performance: {} Wins, {} Loses, {} Ties."
                     )

round_dates = {'Group Stage': ['06/09', '13/09', '04/10', '11/10', '25/10', '01/11'],
               'Round of 16': ['14/02', '07/03'],
               'Quarter-Final': ['11/04', '18/04'],
               'Semi-Final': ['09/05', '16/05'],
               'Final': ['10/06']}
