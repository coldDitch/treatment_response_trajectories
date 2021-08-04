"""
    Functions for plotting the data
"""

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from sklearn.neighbors import KernelDensity
from utils import fit_varnames

az.style.use("arviz-darkgrid")


def plot_samples_grid(fit):
    """Creates a pairplot of posterior samples for one dimensional posterior parameters

    Args:
        fit CmdStanMCMC: the fitted model object
    """
    var_names = fit_varnames(fit)
    az.plot_pair(
        fit, var_names=var_names, divergences=True
    )

def plot_samples(fit):
    """Creates density plots for one dimensional posterior parameters 

    Args:
        fit CmdStanMCMC: the model object fit
    """
    var_names = fit_varnames(fit)
    az.plot_density(
        fit,
        var_names=var_names,
        shade=0.1,
        hdi_prob=1
        ) 



### high level plot functions

def plot_meal_pred(fit, gen_data):
    """plots observed, true and estimated mealtimings, draws arrow from observed to estimated meal timing

    Args:
        fit CmdStanMCMC: the model object fit
        gen_data (dict): dictionary of generated dataset
    """

    if 'meal_reporting_noise' in fit_varnames(fit):
        meal_m = np.median(fit.stan_variable('pred_meals_eiv'), axis=0)
        plt.vlines(gen_data['meal_timing'], 0, 1.2, label='observed meals', color='r')
        plt.vlines(gen_data['true_timing'], 0, 1.1, label='true timing', color='y')
        plt.vlines(meal_m, 0, 1, label='estimated meals', color='b')
        dy = 0
        dx = meal_m - gen_data['meal_timing']
        for i in range(dx.shape[0]):
            plt.arrow(gen_data['meal_timing'][i], 1, dx[i], dy, head_width=0.05, head_length=0.1, length_includes_head=True)
    else:
        plt.vlines(gen_data['true_timing'], 0, 1.1, label='true timing', color='y')
   

def plot_datagen(train_data, test_data=None, label='glucose', marker='x'):
    """[summary]

    Args:
        train_data (dict): Dictionary of train data
        test_data (dict, optional): Dictionary of test data. Defaults to None.
        label (str, optional): Name of the outcome. Defaults to 'glucose'.
        marker (str, optional): How observations ar plotted. Defaults to 'x'.
    """
    plot_glucose(
        train_data['time'],
        train_data['glucose'],
        label, labeladd='train', c='b'+marker)
    if test_data:
        plot_glucose(
            test_data['time'],
            test_data['glucose'],
            label, labeladd='test', c='r'+marker)

def plot_fit(fit):
    """Plots shaded plot of glucose prediction
    and counter factual prediction (prediction if meal was not eaten)

    Args:
        fit CmdStanMCMC: the model object fit
    """
    y = fit.stan_variable('pred_y')
    x = fit.stan_variable('pred_x')[0]
    shadedplot(x, y, label='GP plot')
    y = fit.stan_variable('baseline')
    shadedplot(x, y, label='CF', c='r')

def plot_baseline(fit, gen_data):
    """ plots the baseline and true value

    Args:
        fit CmdStanMCMC: the model object fit
        gen_data (dict): dictionary of generated dataset
    """
    plt.vlines(gen_data['baseline'], 0, 1, label='true baseline', color='r')
    samples = fit.stan_variable('base')
    plot_density(samples, label='baseline estimate')
    #plot_1d_samples(samples)

def plot_response(fit, gen_data):
    """[summary]

    Args:
        fit CmdStanMCMC: the model object fit
        gen_data (dict): dictionary of generated dataset
    """
    plt.plot(gen_data['time'], gen_data['resp'], label='response')
    shadedplot(gen_data['time'], fit.stan_variable('resp'), 'estimate')
    plot_meal_timing(gen_data)


### low level helper functions

def plot_glucose(time, glucose, label, labeladd='', c='b'):
    """[summary]

    Args:
        time (np.array(N)): N number timesteps
        glucose (np.array(N))
        label (str)
        labeladd (str, optional): Defaults to ''.
        c (str, optional): color. Defaults to 'b'.
    """
    plt.plot(time, glucose, c, label=label+' '+labeladd)

def plot_meal_timing(gen_data, c='r'):
    """[summary]

    Args:
        gen_data (dict): dictionary of generated dataset
        c (str, optional): color. Defaults to 'r'.
    """
    time = gen_data['meal_timing']
    height = min(gen_data['glucose']) - 0.5
    plt.vlines(time, -1, height, label='observed meal_timing', color=c)

def shadedplot(x, y_samples, label, c='k'):
    """[summary]

    Args:
        x (np.array(N)): N number of timesteps
        y_samples (np.array(S, N)): S samples, N timesteps
        label (str)
        c (str, optional): color. Defaults to 'k'.
    """
    median = np.quantile(y_samples, 0.5, axis=0)
    lower = np.quantile(y_samples, 0.25, axis=0)
    upper = np.quantile(y_samples, 0.75, axis=0)
    #plt.plot(x, y[:20].T, c='b', alpha=0.2)
    plt.plot(x, median, c=c, label=label)
    plt.fill_between(x, lower, upper, alpha=0.2, color=c)

def estimate_bandwidth(samples):
    """ Approximation for minimizing integrated mse for kernel estimated density

    Args:
        samples (np.array): one dimensional samples

    Returns:
        float: bandwidth estimate
    """
    return np.min((np.std(samples), (np.quantile(samples, 0.75)-np.quantile(samples, 0.25))/1.34)) * 0.9 * np.power(len(samples), -0.2)

def plot_density(samples, label, c='b'):
    """Estimates and plots one dimensional estimate

    Args:
        samples (np.array): one dimensional samples
        label (str)
        c (str, optional): color. Defaults to 'b'.
    """
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
    """[summary]

    Args:
        samples (np.array): one dimensional samples
        m (str, optional): marker color and type. Defaults to '+k'.
        label (str, optional):  Defaults to 'samples'.
    """
    plt.plot(samples, -0.005 - 0.01 *
        np.random.random(samples.shape[0]), m, label=label)

