import pandas as pd
from team_mapping import TEAM_NAME_MAPPING

def save_dataset(df, filename):
    df.to_csv(filename, index=False)
    print(f"Dataset salvo como {filename}")

def normalize_team_name(name):
    return TEAM_NAME_MAPPING.get(name.upper(), name.upper())