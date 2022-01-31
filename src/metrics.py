import numpy as np
import matplotlib.pyplot as plt

def print_metrics(fit, train_data, test_data):
    print(fit.stan_variables().keys())
    print('M1 variance trend:', variance_explained_by_trend(fit))
    print('M2 variance response:', variance_explained_by_response(fit))
    print('M3 mse train:', mse_train(fit, train_data))
    print('M4 mse test:', mse_test(fit, test_data))
    print('M5 abs error in var:', abs_error_response_outcome(fit))


def variance_explained_by_trend(fit):
    varb = fit.stan_variable('train_baseline_gp').mean(axis=0).var()
    varg = fit.stan_variable('train_gluc').mean(axis=0).var()
    return varb/varg

def variance_explained_by_response(fit):
    vary = fit.stan_variable('train_y').mean(axis=0).var()
    varg = fit.stan_variable('train_gluc').mean(axis=0).var()
    return vary/varg - variance_explained_by_trend(fit)

def mse_test(fit, test_data):
    pred = fit.stan_variable('test_gluc').mean(axis=0)
    mse = np.mean(np.square(pred - test_data['glucose']))
    return mse


def mse_train(fit, train_data):
    pred = fit.stan_variable('train_gluc').mean(axis=0)
    mse = np.mean(np.square(pred - train_data['glucose']))
    return mse

def abs_error_response_outcome(fit):
    r = fit.stan_variable('pred_mu').mean(axis=0).var()
    y = fit.stan_variable('train_gluc').mean(axis=0).var()
    return np.abs(r-y)

