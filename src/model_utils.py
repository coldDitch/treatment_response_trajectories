import cmdstanpy
import numpy as np
from config import PARALELLIZE, ALGORITHM, MODELNAME, SEED

def fit_model(data, test_data=None):
    time, glucose, meal_timing = data['measurement_times'], data['y'], data['meals']
    bayesname = MODELNAME
    bayespath = 'stan_models/' + bayesname + '.stan'
    pred_data = {}
    if test_data:
        pred_data['measurement_times'] = np.concatenate((data['measurement_times'], test_data['measurement_times']))
        pred_data['meals'] = np.concatenate((data['meals'], test_data['meals']))
    else:
        ft = 12
        pred_data['measurement_times']=np.linspace(np.min(time), np.max(time) + ft, time.shape[0]*2)
        pred_data['meals']=data['meals']
    print(pred_data.keys())
    dat = {
        'N': time.shape[0],
        'time': time,
        'glucose': glucose,
        'n_meals': meal_timing.shape[0],
        'meal_timing': meal_timing,
        'pred_n': pred_data['measurement_times'].shape[0],
        'pred_times': pred_data['measurement_times'],
        'pred_mn': pred_data['meals'].shape[0],
        'pred_meals': pred_data['meals']
    }
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