import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np


### high level plot functions

def plot_datagen(train_data, test_data=None, gen_data=None, label='glucose'):
    plot_glucose(
        train_data['time'],
        train_data['glucose'],
        label, labeladd='train', c='bx')
    if test_data:
        plot_glucose(
            test_data['time'],
            test_data['glucose'],
            label, labeladd='test', c='rx')
    if gen_data:
        plot_meal_timing(gen_data)

def plot_fit(fit):
    y = fit.stan_variable('pred_y')
    x = fit.stan_variable('pred_x')[0]
    shadedplot(x, y, label='GP plot')
    y = fit.stan_variable('baseline')
    shadedplot(x, y, label='CF', c='r')

def plot_baseline(fit, gen_data):
    plt.vlines(gen_data['baseline'], 0, 1, label='true baseline', color='r')
    samples = fit.stan_variable('base')
    plot_density(samples, label='baseline estimate')
    #plot_1d_samples(samples)

def plot_response(fit, gen_data):
    plt.plot(gen_data['time'], gen_data['resp'], label='response')
    shadedplot(gen_data['time'], fit.stan_variable('resp'), 'estimate')
    plot_meal_timing(gen_data)


### low level helper functions

def plot_glucose(t, glucose, label, labeladd='', c='b'):
    plt.plot(t, glucose, c, label=label+' '+labeladd)

def plot_meal_timing(gen_data, c='r'):
    t = gen_data['meal_timing']
    h = min(gen_data['glucose']) - 0.5
    plt.vlines(t, -1, h, label='true meal_timing', color=c)

def shadedplot(x, y_samples, label, c='k'):
    median = np.quantile(y_samples, 0.5, axis=0)
    lower = np.quantile(y_samples, 0.25, axis=0)
    upper = np.quantile(y_samples, 0.75, axis=0)
    #plt.plot(x, y[:20].T, c='b', alpha=0.2)
    plt.plot(x, median, c=c, label=label)
    plt.fill_between(x, lower, upper, alpha=0.2, color=c)

def estimate_bandwidth(samples):
    """
    approximation for minimizing integrated mse for kernel estimated density
    """
    return np.min((np.std(samples), (np.quantile(samples, 0.75)-np.quantile(samples, 0.25))/1.34)) * 0.9 * np.power(len(samples), -0.2)

def plot_density(samples, label, c='b'):
    bandwidth = estimate_bandwidth(samples)
    # kernel density estimation
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
    samples.reshape(-1, 1))
    X_plot = np.linspace(np.min(samples), np.max(
            samples), 500)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens), '-',
        label=label)
    plt.fill_between(X_plot[:, 0], 0, np.exp(log_dens), alpha=0.2, color=c)

def plot_1d_samples(samples, m='+k', label='samples'):
    plt.plot(samples, -0.005 - 0.01 *
        np.random.random(samples.shape[0]), m, label=label)

