import pandas as pd

df = pd.read_csv('BRA.csv')

# Verifica se existe algum valor de hora na coluna Date
mask = df['Date'].str.contains(r'^\d{1,2}:\d{2}$', na=False)

# Mostra as linhas problemáticas
print(df[mask])
