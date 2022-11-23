"""
    File for running the main experiment
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_utils import plot_individuals, plot_id, plot_meal_timing
from data_generators import generate_data
from model_utils import find_fit
from preprocess import public_data, test_train_split, stan_to_df, df_sorted
from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, SUMMARY, PLOTSAMPLES, SYNTHETIC, TRAIN_PERCENTAGE, PATIENT_ID, NUTRIENTS
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

    print('num meals', test_data['df_meal']['time'].shape)
    print('train meals', train_data['df_meal']['time'].shape)

    #for id in train_data['df_gluc'].id.unique():
    #    plt.figure(figsize=(50, 5))
    #    plt.title('Patient number '+str(id))
    #    plot_id(train_data, id, label='train data',color='r')
    #    plot_id(test_data, id, label='test_data', color='m')
    #    plot_meal_timing(train_data['df_meal'], id)
    #    plot_meal_timing(test_data['df_meal'], id, have_label=False)
    #plt.show()

    fit_gq = find_fit(train_data, test_data)

    #print('lengthscale:', fit_gq.stan_variable('lengthscale').mean(axis=0))
    #print('margstd:', fit_gq.stan_variable('marg_std').mean(axis=0))
    #print('sigma:', fit_gq.stan_variable('sigma').mean(axis=0))
    #print('base:', fit_gq.stan_variable('base').mean(axis=0))
    def print_var(var):
        print(var, np.quantile(fit_gq.stan_variable(var), 0.5, axis=0))

    def print_var_std(var):
        print(var + 'std', np.std(fit_gq.stan_variable(var), axis=0))

    #print('meal reporting noise', fit_gq.stan_variable('meal_reporting_noise').mean(axis=0))

    def response_fun(t):
        lengths = fit_gq.stan_variable('rep_meal_response_lenghts')
        mags = fit_gq.stan_variable('rep_meal_response_magnitudes')

        for i in range(lengths.shape[1]):
            plt.figure()
            y = mags[:100, i].reshape(-1,1) * np.exp(-0.5 * (t - 3 * lengths[:100, i].reshape(-1,1)) ** 2/ lengths[:100, i].reshape(-1,1)**2)
            plt.plot(t, y, c='r')



    print_var('response_magnitude_hier_means')
    print_var('response_magnitude_hier_std')
    print_var('base')
    print_var('sigma')
    print_var('response_magnitude_params_raw')
    print_var('response_length_params')
    print_var('response_const')
    print_var('meal_reporting_noise')
    print_var('meal_reporting_bias')
    print_var('meal_timing_eiv_raw')
    print_var('beta')
    print_var('theta')

    #print_var('rep_meal_response_lenghts')
    #print_var('rep_meal_response_magnitudes')
    #print_var('lengthscale')
    
    #lengths = fit_gq.stan_variable('rep_meal_response_lenghts').mean(axis=0)
    #print('lengths', lengths.size)
    #for nutr in NUTRIENTS:
    #    nutrients = train_data['df_meal'][nutr]
    #    print('nutrients', nutrients.size)
    #    res = stats.linregress(nutrients, lengths)
    #
    #    plt.figure()
    #    plt.scatter(nutrients, lengths)
    #    plt.title(nutr + f"R-squared: {res.rvalue**2:.6f}")
    #plt.show()

    #time_diff = train_data['df_meal']['time'] - fit_gq.stan_variable('meal_timing_eiv').mean(axis=0)
    #print(time_diff.mean())
    #plt.hist(time_diff,bins=50)
    #plt.show()

    #print('starch', fit_gq.stan_variable('starch_magnitude').mean())
    #print('sugar', fit_gq.stan_variable('sugar_magnitude').mean())
    #print('fibr', fit_gq.stan_variable('fibr_magnitude').mean())
    #print('fat', fit_gq.stan_variable('fat_magnitude').mean())
    #print('prot', fit_gq.stan_variable('starch_magnitude').mean())
    result_data = stan_to_df(fit_gq, data)

    print('REP', result_data['df_meal']['rep_magnitude'].mean())
    result_data_train, result_data_test = test_train_split(result_data, train_percentage=TRAIN_PERCENTAGE)

    #plot_individuals(result_data, train_data, test_data)
    print_metrics(result_data_train, result_data_test, train_data, test_data)

    #response_fun(np.arange(-1, 3, 0.1).reshape(1,-1))
    #plt.show()
    #plt.show()


if __name__ == '__main__':
    main()
