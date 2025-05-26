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
    
def get_recent_form(df, team, date, n_games=5):
    mask = ((df['Date'] < date) &
            ((df['Home'] == team) | (df['Away'] == team)))
    recent_games = df.loc[mask].sort_values(by='Date', ascending=False).head(n_games)

    wins = draws = losses = goals_for = goals_against = 0

    for _, match in recent_games.iterrows():
        is_home = match['Home'] == team
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
        else:
            goals_for += match['AG']
            goals_against += match['HG']
            if result == 'A':
                wins += 1
            elif result == 'D':
                draws += 1
            else:
                losses += 1

    return pd.Series({
        'recent_wins': wins,
        'recent_draws': draws,
        'recent_losses': losses,
        'recent_goals_for': goals_for,
        'recent_goals_against': goals_against
    })


# Calculates recent form as win ratio in last n_games
def get_recent_form(df, team, date, n_games=5):
    mask = ((df['Date'] < date) & ((df['Home'] == team) | (df['Away'] == team)))
    recent_games = df.loc[mask].sort_values('Date', ascending=False).head(n_games)
    
    if recent_games.empty:
        return 0.5  # neutral form if no data

    wins = 0
    for _, match in recent_games.iterrows():
        if match['Home'] == team and match['Res'] == 'H':
            wins += 1
        elif match['Away'] == team and match['Res'] == 'A':
            wins += 1
    return wins / n_games

# Calculates goal difference over last n_games
def calculate_goal_diff(df, team, date, n_games=5):
    mask = ((df['Date'] < date) & ((df['Home'] == team) | (df['Away'] == team)))
    recent_games = df.loc[mask].sort_values('Date', ascending=False).head(n_games)

    if recent_games.empty:
        return 0  # neutral if no data

    goal_diff = 0
    for _, match in recent_games.iterrows():
        if match['Home'] == team:
            goal_diff += match['HG'] - match['AG']
        elif match['Away'] == team:
            goal_diff += match['AG'] - match['HG']
    return goal_diff