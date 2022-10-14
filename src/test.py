import pandas as pd


df = pd.read_csv('data/public_dataset.csv')

print(df['id'].unique())