import pandas as pd

df = pd.read_csv('features_brasileirao.csv')
print(df['result'].unique())
print(df['result_enc'].unique())