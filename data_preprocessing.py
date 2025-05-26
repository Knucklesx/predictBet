import pandas as pd
from datetime import timedelta
from utils import normalize_team_name

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df['Home'] = df['Home'].map(normalize_team_name)
    df['Away'] = df['Away'].map(normalize_team_name)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

def get_past_matches(df, team, date, n_years=3):
    start_date = date - timedelta(days=365 * n_years)
    mask = ((df['Date'] < date) &
            ((df['Home'] == team) | (df['Away'] == team)) & 
            (df['Date'] >= start_date))
    return df.loc[mask]

def calculate_team_stats(past_matches, team):
    wins = draws = losses = goals_for = goals_against = 0

    for _, match in past_matches.iterrows():
        is_home = match['Home'] == team
        is_away = match['Away'] == team
        result = match['Res']

        if is_home:
            goals_for += match['HG']
            goals_against += match['AG']
            if result == 'H':
                wins += 1
            elif result == 'D':
                draws += 1
            else:
                losses += 1
        elif is_away:
            goals_for += match['AG']
            goals_against += match['HG']
            if result == 'A':
                wins += 1
            elif result == 'D':
                draws += 1
            else:
                losses += 1

    return pd.Series({
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_for': goals_for,
        'goals_against': goals_against
    })