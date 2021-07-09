import cmdstanpy
import numpy as np
from config import PARALELLISE

def fit_model(data, pred_data=None):
    time, glucose, meal_timing = data['measurement_times'], data['y'], data['meals']
    bayesname = 'TRmodel'
    bayespath = 'stan_models/' + bayesname + '.stan'
    if pred_data==None:
        ft = 12
        pred_data = {}
        pred_data['times']=np.linspace(np.min(time), np.max(time) + ft, time.shape[0]*2)
        pred_data['meals']=data['meals']
    print(pred_data.keys())
    dat = {
        'N': time.shape[0],
        'time': time,
        'glucose': glucose,
        'n_meals': meal_timing.shape[0],
        'meal_timing': meal_timing,
        'pred_n': pred_data['times'].shape[0],
        'pred_times': pred_data['times'],
        'pred_mn': int(pred_data['meals'].shape[0]),
        'pred_meals': pred_data['meals']
    }
    print(dat.keys())
    options = {"STAN_THREADS": True} if PARALELLISE else None
    model = cmdstanpy.CmdStanModel(stan_file=bayespath, cpp_options=options)
    fit = model.sample(data=dat, output_dir='logs', parallel_chains=4)
    return fit