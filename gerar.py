import pandas as pd

df = pd.read_csv('BRA.csv')
home_teams = df['Home'].unique()
away_teams = df['Away'].unique()

all_teams = set(home_teams) | set(away_teams)

for team in sorted(all_teams):
    print(team)
