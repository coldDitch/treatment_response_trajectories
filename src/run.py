"""
    File for running the main experiment
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response, plot_meal_pred, plot_samples_grid, plot_samples, plot_individuals
from data_generators import generate_data
from model_utils import find_fit
from preprocess import public_data, test_train_split, stan_to_df, df_sorted
from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, SUMMARY, PLOTSAMPLES, SYNTHETIC, TRAIN_PERCENTAGE, PATIENT_ID
from metrics import print_metrics


def main():
    if SYNTHETIC:
        data = generate_data(days=DAYS)
    else:
        data = public_data(ids=PATIENT_ID)
    train_data, test_data = test_train_split(data, train_percentage=TRAIN_PERCENTAGE)

    train_data['df_gluc'] = train_data['df_gluc'].sort_values(by=['id','time'])
    test_data['df_gluc'] = test_data['df_gluc'].sort_values(by=['id','time'])

    print('num observations', test_data['df_gluc']['time'].shape)
    print('train observations', train_data['df_gluc']['time'].shape)

    fit_gq = find_fit(train_data, test_data)

    print('lengthscale:', fit_gq.stan_variable('lengthscale').mean(axis=0))
    print('margstd:', fit_gq.stan_variable('marg_std').mean(axis=0))
    print('sigma:', fit_gq.stan_variable('sigma').mean(axis=0))
    print('base:', fit_gq.stan_variable('base').mean(axis=0))
    print('response magnitudes', fit_gq.stan_variable('response_magnitude_params').mean(axis=0))
    print('response lengths', fit_gq.stan_variable('response_length_params').mean(axis=0))
    print('meal reporting noise', fit_gq.stan_variable('meal_reporting_noise').mean(axis=0))

    #print('starch', fit_gq.stan_variable('starch_magnitude').mean())
    #print('sugar', fit_gq.stan_variable('sugar_magnitude').mean())
    #print('fibr', fit_gq.stan_variable('fibr_magnitude').mean())
    #print('fat', fit_gq.stan_variable('fat_magnitude').mean())
    #print('prot', fit_gq.stan_variable('starch_magnitude').mean())
    result_data = stan_to_df(fit_gq, data)
    result_data_train, result_data_test = test_train_split(result_data, train_percentage=TRAIN_PERCENTAGE)


    #plot_individuals(result_data, train_data, test_data)
    print_metrics(result_data_train, result_data_test, train_data, test_data)
    #plt.show()


if __name__ == '__main__':
    main()