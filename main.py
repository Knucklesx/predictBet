from data_preprocessing import load_dataset
from feature_engineering import encode_teams_and_result
from dataset_split import split_dataset
from model_training import train_and_evaluate
from extract_features import extract_features
from utils import save_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = load_dataset('BRA.csv')

le_team = LabelEncoder()
le_team.fit(df['Home'].tolist() + df['Away'].tolist())

features_list = []

for _, row in df.iterrows():
    match_date = row['Date']
    home_team = row['Home']
    away_team = row['Away']

    feature = extract_features(df, home_team, away_team, match_date, le_team)
    feature['result'] = row['Res']
    feature['date'] = match_date
    feature['home_team'] = home_team
    feature['away_team'] = away_team

    features_list.append(feature)

features_df = pd.concat(features_list)
features_df, le_team, le_result = encode_teams_and_result(features_df)

save_dataset(features_df, 'features_brasileirao.csv')

split_dataset()

train_and_evaluate()