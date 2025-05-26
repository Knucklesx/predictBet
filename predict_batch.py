
import pandas as pd
import joblib
from extract_features import extract_features
from team_mapping import TEAM_NAME_MAPPING
from load_dataset import load_dataset
from sklearn.preprocessing import LabelEncoder

def predict_matches(games, model_file='model.pkl', dataset_file='BRA.csv'):
    model = joblib.load(model_file)
    df = load_dataset(dataset_file)

    le_team = LabelEncoder()
    le_team.fit(df['Home'].tolist() + df['Away'].tolist())

    le_result = LabelEncoder()
    le_result.fit(df['Res'].dropna())

    for game in games:
        feature = extract_features(df, game['home_team'], game['away_team'], pd.to_datetime(game['match_date']), le_team)
        pred = model.predict(feature)[0]
        label = le_result.inverse_transform([pred])[0]
        print(f"{game['home_team']} x {game['away_team']} → {label}")