import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from extract_features import extract_features
from team_mapping import TEAM_NAME_MAPPING
from load_dataset import load_dataset

def predict_single_match(model_file='model.pkl', dataset_file='BRA.csv', home_team='Gremio', away_team='Bahia', match_date='2025-05-25'):
    print(f"🔍 Rodando predição para {home_team} x {away_team} em {match_date}")

    model = joblib.load(model_file)
    df = load_dataset(dataset_file)

    le_team = LabelEncoder()
    le_team.fit(df['Home'].tolist() + df['Away'].tolist())

    le_result = LabelEncoder()
    le_result.fit(df['Res'].dropna())

    home_team_std = TEAM_NAME_MAPPING.get(home_team.upper(), home_team.upper())
    away_team_std = TEAM_NAME_MAPPING.get(away_team.upper(), away_team.upper())

    match_date_dt = pd.to_datetime(match_date, errors='coerce')

    feature_input = extract_features(df, home_team_std, away_team_std, match_date_dt, le_team)

    pred = model.predict(feature_input)[0]
    pred_label = le_result.inverse_transform([pred])[0]

    print(f"➡️ Previsão: {pred_label} → ", end='')
    if pred_label == 'H':
        print(f"Vitória do {home_team}")
    elif pred_label == 'D':
        print("Empate")
    else:
        print(f"Vitória do {away_team}")

    return {
        'home_team': home_team,
        'away_team': away_team,
        'match_date': match_date,
        'predicted_result': pred_label
    }
