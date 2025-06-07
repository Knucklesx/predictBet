import pandas as pd
import joblib
from extract_features import extract_features
from team_mapping import TEAM_NAME_MAPPING
from load_dataset import load_dataset
from sklearn.preprocessing import LabelEncoder
import shap
import os

CACHE_FILE = "predictions_cache.csv"

def predict_matches(games, model_file='xgb_model.pkl', dataset_file='BRA.csv', explain=False):
    model = joblib.load(model_file)
    df = load_dataset(dataset_file)

    le_team = LabelEncoder()
    le_team.fit(df['Home'].tolist() + df['Away'].tolist())

    le_result = LabelEncoder()
    le_result.fit(df['Res'].dropna())

    # Load cache
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE)
    else:
        cache = pd.DataFrame(columns=['home_team', 'away_team', 'match_date', 'predicted_result'])

    results = []

    for game in games:
        home_team = TEAM_NAME_MAPPING.get(game['home_team'].upper(), game['home_team'].upper())
        away_team = TEAM_NAME_MAPPING.get(game['away_team'].upper(), game['away_team'].upper())
        match_date = pd.to_datetime(game['match_date'])

        # Check cache
        cached = cache[
            (cache['home_team'] == home_team) &
            (cache['away_team'] == away_team) &
            (cache['match_date'] == str(match_date.date()))
        ]
        if not cached.empty:
            pred_label = cached.iloc[0]['predicted_result']
            print(f"📦 Cached: {home_team} x {away_team} → {pred_label}")
            results.append(cached.iloc[0].to_dict())
            continue

        feature = extract_features(df, home_team, away_team, match_date, le_team)
        pred = model.predict(feature)[0]
        pred_label = le_result.inverse_transform([pred])[0]

        print(f"➡️ {home_team} x {away_team} → {pred_label}")

        if explain:
            explainer = shap.Explainer(model, feature)
            shap_values = explainer(feature)
            top_contribs = sorted(zip(feature.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
            print("🧠 Top reasons:")
            for feat, val in top_contribs:
                print(f"   {feat}: {val:.3f}")

        result = {
            'home_team': home_team,
            'away_team': away_team,
            'match_date': str(match_date.date()),
            'predicted_result': pred_label
        }
        results.append(result)
        cache = pd.concat([cache, pd.DataFrame([result])], ignore_index=True)

    # Save cache
    cache.to_csv(CACHE_FILE, index=False)
    print(f"✅ Results saved to {CACHE_FILE}")

    return results
