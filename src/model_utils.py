"""
    Functions for preparing and fitting the model
"""

import cmdstanpy
import numpy as np
from config import PARALELLIZE, ALGORITHM, MODELNAME, SEED

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
    pred_data = combine_to_prediction_data(data, test_data)
    dat = data.copy()
    dat.update(pred_data)
    inits = {}
    handle_nutrients(dat, test_data)
    handle_eiv(data, test_data, inits)
    options = {"STAN_THREADS": True} if PARALELLIZE else None
    model = cmdstanpy.CmdStanModel(stan_file=bayespath, cpp_options=options)
    if ALGORITHM == 'mcmc':
        fit = model.sample(data=dat,
            output_dir='logs',
            parallel_chains=4,
            threads_per_chain=2,
            show_progress=True,
            seed=SEED,
            inits=inits,
            iter_warmup=1000)
    else:
        raise Exception('not valid inference method')
    fit.save_csvfiles('posterior_data')
    return fit


def combine_to_prediction_data(data, test_data):
    """Combine training and test data to prediction data. A dataset for which we predict the glucose values.

    Args:
        data (dict): dictionary of training data
        test_data (dict): dictionary of test data

    Returns:
        dict: dictionary with times and meals for which we want to predict the glucose
    """
    pred_data = {}
    if test_data:
        pred_data['pred_times'] = np.concatenate((data['time'], test_data['time']))
        pred_data['pred_meals'] = np.concatenate((data['meal_timing'], test_data['meal_timing']))
    else:
        time_horizon = 12
        pred_data['pred_times']=np.linspace(np.min(data['time']), np.max(data['time']) + time_horizon, data['time'].shape[0]*2)
        pred_data['pred_meals']=data['meal_timing']
    pred_data['n_pred'] = pred_data['pred_times'].shape[0]
    pred_data['n_meals_pred'] = pred_data['pred_meals'].shape[0]
    return pred_data

def handle_nutrients(data, test_data):
    """Adds nutrient releated data to the training and prediction set

    Args:
        data (dict): dictionary of training data
        test_data (): dictionary of test data
    """
    data['num_nutrients'] = data['nutrients'].shape[1]
    data['pred_nutrients'] = np.concatenate((data['nutrients'], test_data['nutrients']),axis=0)

def handle_eiv(data, test_data, inits):
    inits['meal_timing_eiv'] = data['meal_timing']
    inits['fut_meal_timing'] = test_data['meal_timing']
