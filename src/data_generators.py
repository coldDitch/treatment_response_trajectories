import cmdstanpy
import numpy as np
from config import SEED

def generate_data(measurements_per_day=96, days=2, lengthscale=1, marg_std=1, baseline=3, response_height=3, response_scale=1):
    bayesname = 'generator'
    bayespath = 'stan_models/' + bayesname + '.stan'
    num_data = int(measurements_per_day * days)
    measurement_times = np.linspace(0, 24 * days, num_data)
    dat = {
        'N': num_data,
        't_meas': measurement_times,
        'marg_std': marg_std,
        'lengthscale': lengthscale,
        'baseline': baseline,
        'a': response_height,
        'b': response_scale,
        'meal_count': 8
    }
    model = cmdstanpy.CmdStanModel(stan_file=bayespath)
    fit = model.sample(data=dat, output_dir='logs', fixed_param=True, chains=1, seed=SEED)
    samples = {key: dat[0] for key, dat in fit.stan_variables().items()}
    samples['measurement_times'] = measurement_times
    samples['baseline'] = np.full(measurement_times.shape, baseline)
    return samples

def test_train_split(samples, train_percentage=0.5):
    dat_len = samples['measurement_times'].shape[0]
    split_len = int(dat_len * train_percentage)
    split_time = samples['measurement_times'][split_len]
    train_samples = {key: dat[:split_len] for key, dat in samples.items()}
    train_samples['meals'] = samples['meals'][samples['meals'] < split_time]
    test_samples = {key: dat[split_len:] for key, dat in samples.items()}
    test_samples['meals'] = samples['meals'][samples['meals'] > split_time]
    return train_samples, test_samples
    
