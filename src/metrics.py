import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import PATIENT_ID

ONLY_MEALS = True

def drop_by_id(dat, id):
    dat['df_gluc'] = dat['df_gluc'][dat['df_gluc'] != id]
    dat['df_meal'] = dat['df_meal'][dat['df_meal'] != id]


def print_metrics(result_data_train, result_data_test, train_data, test_data):
    print('----------')
    pvetrend = variance_explained_by_trend(result_data_train, train_data)
    print('M1 variance trend:', pvetrend[1])
    print(pvetrend[0])
    pveresp = variance_explained_by_response(result_data_train, train_data) 
    print('M2 variance response:', pveresp[1])
    print(pveresp[0])
    msetrain = mse(result_data_train, train_data)
    print('M3 mse train:', msetrain[1])
    print(msetrain[0])
    if ONLY_MEALS:
        filtered_dict(result_data_train)
        filtered_dict(result_data_test)
        filtered_dict(train_data)
        filtered_dict(test_data)
    msetest = mse(result_data_test, test_data)
    print('M4 mse test:', msetest[1])
    print(msetest[0])
    abserrinvar = abs_error_response_outcome(result_data_test, test_data)
    print('M5 abs error in var:', abserrinvar[1])
    print(abserrinvar[0])
    print('----------')
    


def filtered_dict(data):
    mask = meal_windows(data)
    data['df_gluc'] = data['df_gluc'][mask]


def meal_windows(data):
    # filters out the data where there is no meal reported with -1 to +3 hour window 
    times = data['df_gluc']['time']
    mask = np.full((times.shape), False)
    for meal_timing in data['df_meal']['time']:
        mask = mask | (((meal_timing - 60) < times) & (times < (meal_timing + 180)))
    return mask


def variance_explained_by_trend(result_data, train_data):
    pvetrends = []
    for id in PATIENT_ID:
        if id == 9:
            continue
        notzero = result_data['df_gluc']['total_response'] != 0
        mask = (result_data['df_gluc']['id']==id) & notzero
        var_response = result_data['df_gluc']['trend'][mask].var()
        mask = (train_data['df_gluc']['id']==id) & notzero
        var_glucose = train_data['df_gluc']['glucose'][mask].var()
        pvetrends.append(var_response/var_glucose)
    return np.array(pvetrends), np.mean(pvetrends)


def variance_explained_by_response(result_data, train_data):
    pveresp = []
    for id in PATIENT_ID:
        if id == 9:
            continue
        notzero = result_data['df_gluc']['total_response'] != 0
        mask = (result_data['df_gluc']['id']==id) & notzero
        var_clean_response = result_data['df_gluc']['clean_response'][mask].var()
        mask = (train_data['df_gluc']['id']==id) & notzero
        var_glucose = train_data['df_gluc']['glucose'][mask].var()
        pveresp.append(var_clean_response/var_glucose)
    arr, mean = variance_explained_by_trend(result_data, train_data) 
    pveresp = np.array(pveresp) - arr
    return pveresp, np.mean(pveresp)


def mse(result_data, data):
    mse = []
    for id in PATIENT_ID:
        if id == 9:
            continue
        mask = result_data['df_gluc']['id']==id
        y_hat = result_data['df_gluc']['glucose'][mask].values
        mask = data['df_gluc']['id']==id
        y = data['df_gluc']['glucose'][mask].values
        mse.append(np.mean(np.square(y - y_hat)))
    return np.array(mse), np.mean(mse)


def abs_error_response_outcome(result_data, test_data):
    abserr = []
    for id in PATIENT_ID:
        if id == 9:
            continue
        mask = result_data['df_gluc']['id']==id
        var_response = result_data['df_gluc']['total_response'][mask].var()
        mask = test_data['df_gluc']['id']==id
        var_glucose = test_data['df_gluc']['glucose'][mask].var()
        abserr.append(np.abs(var_response - var_glucose))
    return np.array(abserr), np.mean(abserr)

