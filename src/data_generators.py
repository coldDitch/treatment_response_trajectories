import cmdstanpy
import numpy as np
from config import SEED, GENERATORNAME, NUM_NUTRIENTS, GENERATORNAME, MEAL_REPORTING_NOISE, MEAL_REPORTING_BIAS

def generate_data(measurements_per_day=48,
        days=2,
        lengthscale=1,
        marg_std=1,
        baseline=3,
        response_magnitude=5,
        response_length=0.25):
    bayesname = GENERATORNAME
    bayespath = 'stan_models/' + bayesname + '.stan'
    num_data = int(measurements_per_day * days)
    measurement_times = np.linspace(0, 24 * days, num_data)
    dat = {
        'N': num_data,
        'time': measurement_times,
        'marg_std': marg_std,
        'lengthscale': lengthscale,
        'baseline': baseline,
        'response_magnitude': response_magnitude,
        'response_length': response_length,
        'n_meals': 4 * days
    }
    if 'nutrient' in bayesname:
        dat['num_nutrients'] = NUM_NUTRIENTS
        dat['response_magnitude'] = np.full(NUM_NUTRIENTS, response_length)
    if 'eiv' in bayesname:
        dat['meal_reporting_noise'] = MEAL_REPORTING_NOISE
        dat['meal_reporting_bias'] = MEAL_REPORTING_BIAS
    model = cmdstanpy.CmdStanModel(stan_file=bayespath)
    fit = model.sample(data=dat, output_dir='logs', fixed_param=True, chains=1, seed=SEED)
    samples = {key: dat[0] for key, dat in fit.stan_variables().items()}
    samples.update(dat)
    print(samples.keys())
    return samples

def test_train_split(samples, train_percentage=0.8):
    dat_len = samples['time'].shape[0]
    split_len = int(dat_len * train_percentage)
    split_time = samples['time'][split_len]
    train_samples = {key: dat[:split_len] for key, dat in samples.items() if not_intfloat(dat)}
    test_samples = {key: dat[split_len:] for key, dat in samples.items() if not_intfloat(dat)}
    mask = samples['meal_timing'] < split_time
    train_samples['meal_timing'] = samples['meal_timing'][mask]
    test_samples['meal_timing'] = samples['meal_timing'][np.logical_not(mask)]
    if 'nutrient' in GENERATORNAME:
        train_samples['nutrients'] = samples['nutrients'][mask]
        test_samples['nutrients'] = samples['nutrients'][np.logical_not(mask)]
    update_n(train_samples)
    return train_samples, test_samples


def update_n(samples):
    samples['N'] = samples['time'].shape[0]
    samples['n_meals'] = samples['meal_timing'].shape[0]
    
def not_intfloat(x):
    return not (type(x)==int or type(x)==float)