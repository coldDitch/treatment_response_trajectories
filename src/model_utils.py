"""
    Functions for preparing and fitting the model
"""

import os
import cmdstanpy
import numpy as np
from config import PARALELLIZE, ALGORITHM, MODELNAME, SEED, TRAIN_PERCENTAGE, PATIENT_ID, PRIOROVERRESPONSEH
from preprocess import data_to_stan

def name_run():
    list_of_params = [MODELNAME, PATIENT_ID, TRAIN_PERCENTAGE, SEED, PRIOROVERRESPONSEH]
    list_of_params = [str(el) for el in list_of_params]
    name = '_'.join(list_of_params)
    return name

def find_fit(data, test_data=None):
    run_path = 'posterior_data/'+name_run()
    if not os.path.isdir(run_path):
        print('new run')
        fit_model(data, test_data)
    else:
        print('previous fit found')
    return generated_quantities(data, test_data)

def model_files():
    posterior_path = "posterior_data/"+name_run()+'/'
    dirs = os.listdir(posterior_path)
    last_fit = sorted(dirs)[-4:]
    last_fit = [posterior_path + l for l in last_fit]
    return last_fit

def generated_quantities(data, test_data=None):
    bayesname = MODELNAME
    bayespath = 'stan_models/' + bayesname + '_gq.stan'
    model = cmdstanpy.CmdStanModel(stan_file=bayespath)
    train_data, pred_data = data_to_stan(data, test_data)
    last_fit = model_files()
    fit = model.generate_quantities(data=pred_data,
            mcmc_sample=last_fit,
            gq_output_dir='logs')
    return fit

def fit_model(data, test_data=None):
    """Fits the model to the data and predicts glucose values to test_data

    Args:
        data (dict): dictionary containing the training data,
            parameter values are inferred based on these observations
        test_data (dict, optional): predictions are . Defaults to None.

    Raises:
        Exception: if not valid inference method was chosen

    Returns:
        CmdStanMCMC: returns object which contains the model fit
    """
    bayesname = MODELNAME
    bayespath = 'stan_models/' + bayesname + '.stan'
    train_data, pred_data = data_to_stan(data, test_data)
    inits = {}
    handle_eiv(train_data, pred_data, inits)
    options = {"STAN_THREADS": True} if PARALELLIZE else None
    model = cmdstanpy.CmdStanModel(stan_file=bayespath, cpp_options=options)
    if ALGORITHM == 'mcmc':
        fit = model.sample(data=train_data,
            output_dir='logs',
            parallel_chains=4,
            threads_per_chain=2,
            show_progress=True,
            seed=SEED,
            inits=inits,
            iter_warmup=1000,
            refresh=1)
    else:
        raise Exception('not valid inference method')
    fit.save_csvfiles('posterior_data/' + name_run())
    return fit




def handle_eiv(data, test_data, inits):
    inits['lengthscale'] = np.full(data['num_ind'], 3)
    inits['marg_std'] = np.full(data['num_ind'], 1)
    inits['sigma'] = np.full(data['num_ind'], 0.5)
    inits['base'] = np.full(data['num_ind'], 5)
    inits['response_magnitude_params'] = np.full((data['num_ind'], data['num_nutrients']), 5)
    inits['response_length_params'] = np.full((data['num_ind'], data['num_nutrients']), 0.5)
    inits['meal_reporting_noise'] = np.full(data['num_ind'], 0.5)
    inits['meal_timing_eiv'] = data['meal_timing']
