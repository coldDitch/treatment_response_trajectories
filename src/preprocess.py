import pandas as pd
import numpy as np
from data_generators import update_n

def minmax_scale(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def public_data(id=0):
    dat = {}
    nutrients = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']
    df = pd.read_csv('data/public_dataset.csv')
    for nutrient in nutrients:
        df[nutrient] = minmax_scale(df[nutrient])
    df = df[df['id'] == id]
    df['time'] = df['time']/60
    mask = df[nutrients].notna().any(1)
    df_meal = df.loc[mask, nutrients + ['time', 'id']]
    df = df[df['glucose'].notna()]
    dat['glucose'] = df['glucose'].values
    dat['time'] = df['time'].values
    dat['meal_timing'] = df_meal['time'].values
    dat['nutrients'] = df_meal[nutrients].values
    update_n(dat)
    return dat
