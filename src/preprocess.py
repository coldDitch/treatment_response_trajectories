import pandas as pd
import numpy as np
from config import PRIOROVERRESPONSEH, NUTRIENTS



def stan_to_df(gq, data):
    df_gluc = data['df_gluc'].copy()
    df_meal = data['df_meal'].copy()
    gluc_samples = gq.stan_variable('glucose')
    df_gluc['glucose']  = np.quantile(gq.stan_variable('glucose'), 0.5, axis=0)
    df_gluc['q25']  = np.quantile(gq.stan_variable('glucose'), 0.025, axis=0)
    df_gluc['q75']  = np.quantile(gq.stan_variable('glucose'), 0.975, axis=0)
    df_gluc['trend'] = gq.stan_variable('trend').mean(axis=0)
    df_gluc['total_response'] = gq.stan_variable('total_response').mean(axis=0)
    resp_samples = gq.stan_variable('total_response')
    df_gluc['clean_response'] = gq.stan_variable('clean_response').mean(axis=0)
    df_gluc['log_density'] = gq.stan_variable('SAP').mean(axis=0)
    df_meal['rep_timing'] = gq.stan_variable('rep_meal_response_timings').mean(axis=0)
    df_meal['rep_length'] = gq.stan_variable('rep_meal_response_lenghts').mean(axis=0)
    df_meal['rep_magnitude'] = gq.stan_variable('rep_meal_response_magnitudes').mean(axis=0)

    return {'df_gluc': df_gluc, 'df_meal': df_meal, 'gluc_samples': gluc_samples, 'resp_samples': resp_samples}

def data_to_stan(train_data, test_data):
    pred_data = combine_to_prediction_data(train_data, test_data)
    train_data = create_ind_matrix(train_data)
    pred_data = create_ind_matrix(pred_data)
    train_data = dfs_to_dict(train_data)
    pred_data = dfs_to_dict(pred_data)
    pred_data['use_prior'] = int(PRIOROVERRESPONSEH) 
    train_data['use_prior'] = int(PRIOROVERRESPONSEH) 

    ## add approximation configuration parameters

    train_data['L'] = 5/2*pred_data['time'].max()
    train_data['M'] = 200

    pred_data['L'] = 5/2*pred_data['time'].max()
    pred_data['M'] = 200

    train_data['lengthscale'] = 120
    train_data['marg_std'] = 1
    
    pred_data['lengthscale'] = 120
    pred_data['marg_std'] = 1
    


    pred_data['n_train_meals'] = len(train_data['meal_timing'])
    pred_data['num_train_meals_ind'] = train_data['num_meals_ind']
    pred_data['num_train_gluc_ind'] = train_data['num_gluc_ind']
    pred_data['ind_idx_train_meals'] = train_data['ind_idx_meals']
    update_n(train_data)
    update_n(pred_data)
    pred_data['N_train'] = train_data['N']
    pred_data['glucose_train'] = pred_data['glucose']

    return train_data, pred_data

def dfs_to_dict(data):
    data['id'] = data['df_gluc']['id'].values
    data['glucose'] = data['df_gluc']['glucose'].values
    data['time'] = data['df_gluc']['time'].values
    data['meal_timing'] = data['df_meal']['time'].values
    data['nutrients'] = data['df_meal'][NUTRIENTS].values
    return data


def minmax_scale(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def logistic_scale(x):
    return 1/(1+np.exp(-x))

def unit_scaled(x):
    return x/(1+x)

def result_data_split(data, train_percentage=0.8):
    train, test = test_train_split(data, train_percentage)
    df_gluc = data['df_gluc']
    train_gluc_samples = 
    ids = pd.unique(df_gluc['id'])
    for id in ids:
        df_gluc_ind = df_gluc[df_gluc['id']==id]
        time_indx = int(len(df_gluc_ind['time']) * train_percentage)
        time_cutoff = df_gluc_ind.iloc[time_indx]['time']
        gluc_mask = df_gluc_ind['time'] <= time_cutoff
        data['gluc_samples'][:, time_indx]




def test_train_split(data, train_percentage=0.8):
    df_gluc_train = pd.DataFrame()
    df_gluc_test = pd.DataFrame()
    df_meal_train = pd.DataFrame()
    df_meal_test = pd.DataFrame()
    df_gluc = data['df_gluc']
    df_meal = data['df_meal']
    ids = pd.unique(df_gluc['id'])
    for id in ids:
        df_gluc_ind = df_gluc[df_gluc['id']==id]
        df_meal_ind = df_meal[df_meal['id']==id]
        time_indx = int(len(df_gluc_ind['time']) * train_percentage)
        time_cutoff = df_gluc_ind.iloc[time_indx]['time']
        gluc_mask = df_gluc_ind['time'] <= time_cutoff
        meal_mask = df_meal_ind['time'] <= time_cutoff
        df_gluc_train = pd.concat([df_gluc_train, df_gluc_ind[gluc_mask]])
        df_gluc_test = pd.concat([df_gluc_test, df_gluc_ind[~gluc_mask]])
        df_meal_train = pd.concat([df_meal_train, df_meal_ind[meal_mask]])
        df_meal_test = pd.concat([df_meal_test, df_meal_ind[~meal_mask]])
    train = {'df_gluc': df_gluc_train, 'df_meal': df_meal_train}
    test = {'df_gluc': df_gluc_test, 'df_meal': df_meal_test}
    return train, test


def public_data(ids=[1]):
    nutrients = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']
    df = pd.read_csv('data/public_dataset.csv')
    for nutrient in nutrients:
        df[nutrient] /= np.mean(df[nutrient], axis=0)
        df[nutrient] = unit_scaled(df[nutrient])
    df = df[df['id'].isin(ids)]
    time_mean = df['time'].mean()
    df['time'] = df['time'] - time_mean 
    mask = df[nutrients].notna().any(1) & (df[nutrients] > 0.1).any(1)
    df_meal = df.loc[mask, nutrients + ['time', 'id']]
    df_meal = df_meal.rename({'time': 'meal_timing'})
    df = df[df['glucose'].notna()]
    dat = {'df_gluc': df, 'df_meal':df_meal}
    dat['df_gluc'] = df_sorted(dat['df_gluc'])
    return dat


def process_to_stan(dat):
    return dat


def public_data2(ids=[0]):
    dat = {}
    nutrients = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']
    df = pd.read_csv('data/public_dataset.csv')
    for nutrient in nutrients:
        df[nutrient] = minmax_scale(df[nutrient])
    df = df[df['id'].isin(ids)]
    df['time'] = df['time']/60
    print('small filter', (df[nutrients] > 0).any(0))
    mask = df[nutrients].notna().any(1)
    df_meal = df.loc[mask, nutrients + ['time', 'id']]
    df = df[df['glucose'].notna()]
    dat['num_ind'] = len(ids)
    dat['glucose'] = df['glucose'].values
    dat['time'] = df['time'].values
    dat['meal_timing'] = df_meal['time'].values
    dat['nutrients'] = df_meal[nutrients].values

def create_ind_matrix(dat):
    ids = dat['df_gluc']['id'].unique()
    dat['num_ind'] = len(ids)
    dat['ind_idx_gluc'], dat['num_gluc_ind'] = ind_matrix(dat['df_gluc']['id'].values,ids)
    dat['ind_idx_meals'], dat['num_meals_ind'] = ind_matrix(dat['df_meal']['id'].values,ids)
    return dat


def update_n(data):
    """Updates the variables describing array lengths to correspond with the actual lengths

    Args:
        samples (dict): 
    """
    data['N'] = len(data['time'])
    data['num_nutrients'] = data['nutrients'].shape[1]
    if 'pred_meals' in data.keys():
        data['n_meals_pred'] = len(data['pred_meals'])
    if 'meal_timing' in data.keys():
        data['n_meals'] = len(data['meal_timing'])


def combine_to_prediction_data(train_data, test_data):
    """Combine training and test data to prediction data. A dataset for which we predict the glucose values.

    Args:
        data (dict): dictionary of training data
        test_data (dict): dictionary of test data

    Returns:
        dict: dictionary with times and meals for which we want to predict the glucose
    """
    pred_data = {}
    pred_data['df_gluc'] = pd.concat((train_data['df_gluc'], test_data['df_gluc']))
    pred_data['df_meal'] = pd.concat((train_data['df_meal'], test_data['df_meal']))
    pred_data['df_gluc'] = df_sorted(pred_data['df_gluc']) 
    return pred_data


def handle_nutrients(data, test_data):
    """Adds nutrient releated data to the training and prediction set

    Args:
        data (dict): dictionary of training data
        test_data (): dictionary of test data
    """
    data['num_nutrients'] = data['nutrients'].shape[1]
    data['pred_nutrients'] = np.concatenate((data['nutrients'], test_data['nutrients']),axis=0)


def ind_matrix(x, id_range):
    arg_lst = [np.argwhere(x==id) for id in id_range]
    num_ind = [len(arg) for arg in arg_lst]
    arg_lst = [np.pad(arg.ravel(), (0, len(x)-len(arg))).reshape(-1,1) for arg in arg_lst]
    return np.concatenate(arg_lst, axis=1).T + 1, np.array(num_ind)

def df_sorted(df):
    return df.sort_values(by=['id','time'])
