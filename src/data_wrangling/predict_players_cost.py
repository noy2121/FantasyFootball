import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)


# Preprocess the data
def preprocess_data(df, encoding_method='onehot'):

    # Select relevant features
    features = [
        'position',
        'goals',
        'assists',
        'lineups',
        'club_id'
    ]

    # Separate features and target
    X = df[features]
    y = df['2023/24_cost']

    numeric_features = [
        'goals',
        'assists',
        'lineups',
        'club_id'
    ]
    categorical_features = ['position']

    # Create preprocessor
    if encoding_method == 'onehot':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
    else:  # label encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('label', LabelEncoder())
        ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor


# Train the model
def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    return model


# Predict costs for previous seasons
def predict_previous_seasons(model, previous_seasons_data, data_dir):

    for season, data in previous_seasons_data.items():
        if season == '2023/24':
            continue
        features = [
            'position',
            'goals',
            'assists',
            'lineups',
            'club_id',
        ]

        X = data[features]
        predicted_costs = model.predict(X)
        data[f'{season}_cost'] = np.round(predicted_costs)

        # Save predictions
        data.to_csv(f'{data_dir}/cost_predictions/{season[:-3]}_season_with_predicted_costs.csv', index=False)
        print(f"Saved predictions for {season} season.")


def rename_columns(df, season):
    return df.rename(columns={f'{season}_total_goals': 'goals',
                              f'{season}_total_assists': 'assists',
                              f'{season}_lineups': 'lineups',
                              f'{season}_club_id': 'club_id'})


def splits_to_seasons(df, seasons):
    dfs_per_seasons = {}
    for season in seasons:
        cols = [
            'player_name',
            'position',
            f'{season}_total_goals',
            f'{season}_total_assists',
            f'{season}_lineups',
            f'{season}_club_id',
            '2023/24_cost'
            ]
        curr_df = df[cols]
        dfs_per_seasons[season] = rename_columns(curr_df, season)

    return dfs_per_seasons


# Main function
def main():
    # Load current season data (with known costs)
    data_dir = r'C:\Users\Noy\PycharmProjects\FantasyFootball\data'
    players_df = pd.read_csv(f'{data_dir}/csvs/players_with_cost.csv')

    # Preprocess current season data
    seasons = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22', '2022/23', '2023/24']
    dfs_per_season = splits_to_seasons(players_df, seasons)
    X, y, preprocessor = preprocess_data(dfs_per_season['2023/24'])

    # Train the model
    model = train_model(X, y, preprocessor)

    # Predict costs for previous seasons
    predict_previous_seasons(model, dfs_per_season, data_dir)


if __name__ == "__main__":
    main()
