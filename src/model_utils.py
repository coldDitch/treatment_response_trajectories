import cmdstanpy
import numpy as np
from config import PARALELLIZE, ALGORITHM, MODELNAME, SEED

def fit_model(data, test_data=None):
    bayesname = MODELNAME
    bayespath = 'stan_models/' + bayesname + '.stan'
    pred_data = combine_to_prediction_data(data, test_data)
    dat = data.copy()
    dat.update(pred_data)
    if 'nutrient' in bayesname:
        handle_nutrients(dat, test_data)
    options = {"STAN_THREADS": True} if PARALELLIZE else None
    model = cmdstanpy.CmdStanModel(stan_file=bayespath, cpp_options=options)
    if ALGORITHM == 'mcmc':
        fit = model.sample(data=dat, output_dir='logs', parallel_chains=4, show_progress=True, seed=SEED)
    elif ALGORITHM == 'vi':
        fit = model.variational(data=dat, output_dir='logs')
    elif ALGORITHM == 'mle':
        fit = model.optimize(data=dat, output_dir='logs')
    else:
        raise Exception('not valid inference method')
    return fit

def combine_to_prediction_data(data, test_data):
    pred_data = {}
    if test_data:
        pred_data['pred_times'] = np.concatenate((data['time'], test_data['time']))
        pred_data['pred_meals'] = np.concatenate((data['meal_timing'], test_data['meal_timing']))
    else:
        ft = 12
        pred_data['pred_times']=np.linspace(np.min(time), np.max(time) + ft, time.shape[0]*2)
        pred_data['pred_meals']=data['meal_timing']
    pred_data['n_pred'] = pred_data['pred_times'].shape[0]
    pred_data['n_meals_pred'] = pred_data['pred_meals'].shape[0]
    return pred_data

def handle_nutrients(dat, test_data):
    dat['num_nutrients'] = dat['nutrients'].shape[1]
    dat['pred_nutrients'] = np.concatenate((dat['nutrients'], test_data['nutrients']),axis=0)
