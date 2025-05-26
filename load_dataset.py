import pandas as pd
from team_mapping import TEAM_NAME_MAPPING

def load_dataset(filepath='BRA.csv'):
    df = pd.read_csv(filepath)
    df['Home'] = df['Home'].str.upper().map(lambda x: TEAM_NAME_MAPPING.get(x, x))
    df['Away'] = df['Away'].str.upper().map(lambda x: TEAM_NAME_MAPPING.get(x, x))
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    return df
