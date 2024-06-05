
import pandas as pd


class FootballDataset:

    def __init__(self, name, dfs):

        self.dfs = dfs
        self.df = dfs[name]
        self.name = name
        self.columns = self.df.columns.tolist()

    def _create_text_df_for_valuations(self):
        """
        first name: fname, lname: last_name date: date, market value in euro: market_value_in_euro, club: club
        """
        club_map = self.dfs['clubs'].set_index('club_id')['name'].to_dict()
        rows = []
        for i, row in self.df.iterrows():
            pid = row['player_id']

            first_name = self.dfs['players'].loc[self.dfs['players']['player_id'] == pid]['first_name'].values[0]
            last_name = self.dfs['players'].loc[self.dfs['players']['player_id'] == pid]['last_name'].values[0]

            d = {
                'first_name': first_name,
                'last_name': last_name,
                'date': row['date'],
                'market_value_in_euro': row['market_value_in_eur'],
                'club': club_map[row['current_club_id']]
            }
            text = ', '.join([f'{" ".join(k.split("_"))}: {v}' for k, v in d.items()])
            rows.append(text)

        text_df = pd.DataFrame(data=rows, columns=['text'])

        return text_df

    def _create_text_df_for_players(self):
        """
        name: full name, last_season: ls, current_club_name: ccn, country_of_birth: cb, country_of_citizenship: cc,
        date_of_birth: dob, position: p, sub_position: sp, foot: f, height_in_cm, market_value_in_euro: m,
        highest_market_value_in_euro: hm
        """

        rows = []
        for i, row in self.df.iterrows():
            d = {
                'name': row['name'],
                'last_season': row['last_season'],
                'current_club_name': row['current_club_name'],
                'country_of_birth': row['country_of_birth'],
                'country_of_citizenship': row['country_of_citizenship'],
                'date_of_birth': row['date_of_birth'],
                'position': row['position'],
                'sub_position': row['sub_position'],
                'foot': row['foot'],
                'height_in_cm': row['height_in_cm'],
                'market_value_in_euro': row['market_value_in_eur'],
                'highest_market_value_in_euro': row['highest_market_value_in_eur']
            }
            text = ', '.join([f'{" ".join(k.split("_"))}: {v}' for k, v in d.items()])
            rows.append(text)

        text_df = pd.DataFrame(data=rows, columns=['text'])

        return text_df

    def _create_text_df_for_appearances(self):
        """
        a bit tricky, need more work
        """
        pass

    def _create_text_df_for_games(self):
        """
        competition: c_name, season: s, round: r, date: date, home club: home club name, away club: away club name,
        home goals: hg, away goals: ag, home club position: hcp, away club position: acp, home club manager: hcm,
        away club manager: acm, stadium: s, attendance: a, referee: ref, home club formation: hcf,
        away club formation: acf, competition type: ct
        """

        competition_map = self.dfs['competitions'].set_index('competition_id')['name'].to_dict()
        rows = []
        for i, row in self.df.iterrows():
            d = {
                'competition_id': competition_map[row['competition_id']],
                'competition_type': row['competition_type'],
                'season': row['season'],
                'round': row['round'],
                'date': row['date'],
                'home_club_name': row['home_club_name'],
                'away_club_name': row['away_club_name'],
                'home_club_goals': row['home_club_goals'],
                'away_club_goals': row['away_club_goals'],
                'home_club_position': row['home_club_position'],
                'away_club_position': row['away_club_position'],
                'home_club_manager_name': row['home_club_manager_name'],
                'away_club_manager_name': row['away_club_manager_name'],
                'stadium': row['stadium'],
                'attendance': row['attendance'],
                'referee': row['referee'],
                'home_club_formation': row['home_club_formation'],
                'away_club_formation': row['away_club_formation']
            }
            text = ', '.join([f'{" ".join(k.split("_"))}: {v}' for k, v in d.items()])
            rows.append(text)

        text_df = pd.DataFrame(data=rows, columns=['text'])

        return text_df

    def create_text_df(self):
        """
        create a text_df for each df with shape (n_rows, 1) containing only text, so we could fine tune the
        finetune on sentences and not tabular data

        :return: pd.DataFrame in the corresponding format
        """

        if self.name == 'player_valuations':
            return self._create_text_df_for_valuations()
        elif self.name == 'players':
            return self._create_text_df_for_players()
        elif self.name == 'appearances':
            return self._create_text_df_for_appearances()
        elif self.name == 'games':
            return self._create_text_df_for_games()

    def __str__(self):
        return f'{self.name}\n{self.df.head(10)}'

    def __len__(self):
        return len(self.df.shape[0])

