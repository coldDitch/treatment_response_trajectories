import cmdstanpy
import json
import numpy as np
from config import SEED, GENERATORNAME, GENERATORNAME, LOGS

def generate_data(measurements_per_day=96, days=2):
    """Uses stan model to draw samples from data generating distribution.
    One samples is drawn from the distribution to form a dataset.

    Args:
        measurements_per_day (int, optional): [description]. Defaults to 96.
        days (int, optional): [description]. Defaults to 2.
        lengthscale (float, optional): [description]. Defaults to 0.5.
        marg_std (float, optional): [description]. Defaults to 0.5.
        baseline (float, optional): [description]. Defaults to 4.5.
        response_magnitude (float, optional): [description]. Defaults to 4.
        response_length (float, optional): [description]. Defaults to 0.5.

    Returns:
        dict: dictionary where key is name of modeling variable and value is corresponding generated data
    """
    bayesname = GENERATORNAME
    bayespath = 'stan_models/' + bayesname + '.stan'
    num_data = int(measurements_per_day * days)
    measurement_times = np.linspace(0, 24 * days, num_data)
    with open('generator.json') as json_file:
        dat = json.loads(json_file.read())
    dat['time'] = measurement_times
    update_n(dat)
    model = cmdstanpy.CmdStanModel(stan_file=bayespath)
    fit = model.sample(data=dat, output_dir=LOGS, fixed_param=True, chains=1, seed=SEED)
    gen_data = {key: dat[0] for key, dat in fit.stan_variables().items()}
    gen_data.update(dat)
    return gen_data

def test_train_split(data, train_percentage=0.8):
    """Splits data into test and training set

    Args:
        data (dict): dictionary where key is name of modeling variable and value is corresponding generated data
        train_percentage (float, optional): [description]. Defaults to 0.8.

    Returns:
        dict, dict : data split in two
    """
    dat_len = data['time'].shape[0]
    split_len = int(dat_len * train_percentage)
    split_time = data['time'][split_len]
    # TODO this splits all data at split_len and then other data is handled seperately
    # for this to work consistently stan model should be reformated such that dimension of all matrices / vectors
    # in generated quatities block are [N, other dims] where N is number of glucose measurements
    train_data = {key: dat[:split_len] for key, dat in data.items() if not_intfloat(dat)}
    test_data = {key: dat[split_len:] for key, dat in data.items() if not_intfloat(dat)}
    mask = data['meal_timing'] < split_time
    train_data['meal_timing'] = data['meal_timing'][mask]
    test_data['meal_timing'] = data['meal_timing'][np.logical_not(mask)]
    train_data['nutrients'] = data['nutrients'][mask]
    test_data['nutrients'] = data['nutrients'][np.logical_not(mask)]
    update_n(train_data)
    return train_data, test_data


def update_n(data):
    """Updates the variables describing array lengths to correspond with the actual lengths

    Args:
        samples (dict): 
    """
    data['N'] = len(data['time'])
    data['num_nutrients'] = len(data['response_magnitude_params'])
    if 'meal_timing' in data.keys():
        data['n_meals'] = len(data['meal_timing'])

def not_intfloat(test_el):
    """tests if data x is int or float

    Args:
        x (?): any data

    Returns:
        boolean: true if not int or float
    """
    return not (type(test_el)==int or type(test_el)==float)
