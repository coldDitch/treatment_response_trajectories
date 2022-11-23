import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping

from config import PATIENT_ID

ONLY_MEALS = False

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
    abserrinvar = abs_expected_error_in_var(result_data_test, test_data)
    print('M5 abs error in var:', abserrinvar[1])
    print(abserrinvar[0])
    avglogpreddens = avg_log_pred_density(result_data_test)
    print('M6 avg log pred density', avglogpreddens[1])
    print(avglogpreddens[0])
    print('----------')
    #eiverrs = eiv_free_mse(result_data_test, test_data)
    #print('M6 mse in meal timings', eiverrs[1])
    #print(eiverrs[0])
    #print('M7 mse without errors in timings', eiverrs[3])
    #print(eiverrs[2])
    

def avg_log_pred_density(result_data_test):
    avg_densities = []
    for id in PATIENT_ID:
        mask = result_data_test['df_gluc'].id == id
        avg_densities.append(np.mean(result_data_test['df_gluc'][mask].log_density))
    return avg_densities, np.mean(avg_densities)

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
        mask = result_data['df_gluc']['id']==id
        y_hat = result_data['df_gluc']['glucose'][mask].values
        mask = data['df_gluc']['id']==id
        y = data['df_gluc']['glucose'][mask].values
        mse.append(np.mean(np.square(y - y_hat)))
    return np.array(mse), np.mean(mse)


def abs_error_response_outcome(result_data, test_data):
    abserr = []
    for id in PATIENT_ID:
        mask = result_data['df_gluc']['id']==id
        var_response = result_data['df_gluc']['total_response'][mask].var()
        mask = test_data['df_gluc']['id']==id
        var_glucose = test_data['df_gluc']['glucose'][mask].var()
        abserr.append(np.abs(var_response - var_glucose))
    return np.array(abserr), np.mean(abserr)

def abs_expected_error_in_var(result_data, test_data):
    abserr = []
    for id in PATIENT_ID:
        mask = result_data['df_gluc']['id']==id
        var_response = result_data['gluc_samples'][:,mask].var(axis=1)
        print(var_response.shape)
        var_response = var_response.mean()
        mask = test_data['df_gluc']['id']==id
        var_glucose = test_data['df_gluc']['glucose'][mask].var()
        abserr.append(np.abs(var_response - var_glucose))
    return np.array(abserr), np.mean(abserr)


def response(heights, lengths, time, meal_timing):
    resps = heights.reshape(1,-1) * np.exp(-0.5 * (time.reshape(-1,1)-meal_timing.reshape(1,-1))**2 / lengths.reshape(1,-1) ** 2)
    return np.sum(resps, axis=1)


def eiv_free_mse(result_data_test, test_data):
    eivfmse = []
    eivtmse = []
    for i in PATIENT_ID:
        print('eivfree id:', i)
        gluc_mask = test_data['df_gluc'].id == i
        meal_mask = test_data['df_meal'].id == i
        df_meal = result_data_test['df_meal'][meal_mask]
        df_gluc = result_data_test['df_gluc'][gluc_mask]
        h = df_meal.rep_magnitude.values
        l = df_meal.rep_length.values
        t = df_meal.rep_timing.values
        y = test_data['df_gluc'][gluc_mask]['glucose']

        def mse(meal_timing_error):
            y_hat = df_gluc.trend + response(h,l,df_gluc.time.values, t+meal_timing_error)
            return np.mean((y-y_hat)**2) + np.mean((meal_timing_error/120)**2)
        bounds =[(-400, 400) for i in range(df_meal.id.shape[0])]
        x0 = np.zeros(df_meal.id.shape[0])
        res = basinhopping(mse, x0=x0, stepsize=200, T=100)
        print(res.x)
        y_hat_est = df_gluc.trend + response(h,l,df_gluc.time.values, t+res.x)
        y_hat = df_gluc.trend + response(h,l,df_gluc.time.values, t)

        time = test_data['df_gluc'][gluc_mask].time
        plt.figure()
        plt.scatter(time, y)
        plt.plot(time, y_hat)
        plt.plot(time, y_hat_est)
        eivtmse.append(np.mean((res.x)**2))
        eivfmse.append(np.mean((y-y_hat_est)**2))
        plt.show()
    return eivtmse, np.mean(eivtmse), eivfmse, np.mean(eivfmse)
