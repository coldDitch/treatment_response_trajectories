import matplotlib.pyplot as plt
import numpy as np


def plot_datagen(data):
    t = data['measurement_times']
    y = data['y']
    response = data['cumulative_res']
    noise = data['base_variation'] + data['baseline']
    plt.plot(t, y, 'rx', label='glucose',)
    plt.plot(t, response, 'y', label='response')
    plt.vlines(data['meals'],0, min(response)-1, label='true meals')

def plot_fit(fit):
    y = fit.stan_variable('pred_y')
    x = fit.stan_variable('pred_x')[0]
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    #plt.plot(x, y[:20].T, c='b', alpha=0.2)
    plt.plot(x, mean, c='k', label='GP predict')
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, color='k')
