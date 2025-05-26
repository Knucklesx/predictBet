from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

def encode_teams_and_result(df):
    le_team = LabelEncoder()
    le_team.fit(df['home_team'].tolist() + df['away_team'].tolist())

    df['home_team_enc'] = le_team.transform(df['home_team'])
    df['away_team_enc'] = le_team.transform(df['away_team'])

    le_result = LabelEncoder()
    df['result_enc'] = le_result.fit_transform(df['result'])

    return df, le_team, le_result

def head_to_head_stats(df, home_team, away_team, date, n_years=3):
    start_date = date - timedelta(days=365 * n_years)
    mask = (
        (df['Date'] < date) &
        (df['Date'] >= start_date) &
        ((df['Home'] == home_team) | (df['Away'] == away_team))
    )
    h2h = df.loc[mask]
    if h2h.empty:
        return 0.5

    home_wins = sum((h2h['Home'] == home_team) & (h2h['Res'] == 'H')) + \
                sum((h2h['Away'] == home_team) & (h2h['Res'] == 'A'))
    return home_wins / len(h2h)