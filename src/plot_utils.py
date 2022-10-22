"""
    Functions for plotting the data
"""

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from sklearn.neighbors import KernelDensity
from utils import fit_varnames
from config import PATIENT_ID

az.style.use("arviz-darkgrid")


def plot_individuals(result_data, train_data, test_data):
    for id in PATIENT_ID:
        plt.figure(figsize=(50, 5))
        plt.title('Patient number '+str(id))
        mask = result_data['df_gluc']['id'] == id
        plt.plot(result_data['df_gluc']['time'][mask], result_data['df_gluc']['trend'][mask], color='y')
        plot_line(result_data, id, label='model fit', color='b')
        plot_traces(result_data, id )
        plot_quantiles(result_data, id, color='b')
        plot_id(train_data, id, label='train data',color='r')
        plot_id(test_data, id, label='test_data', color='m')
        plot_meal_timing(train_data['df_meal'], id)
        plot_meal_timing(test_data['df_meal'], id, have_label=False)
        plt.legend()
        plt.savefig('../figures/patient'+str(id))

def plot_traces(data, id, label='', color='y'):
    mask = data['df_gluc']['id']==id
    resp = data['resp_samples'][:100, mask].T
    plt.plot(data['df_gluc'][mask]['time'], resp, color, alpha=0.2)

def plot_glucose_sample_trajectories(result_data, id, color):
    mask = result_data['df_gluc']['id']==id
    df = result_data['df_gluc'][mask]
    plt.plot(df['time'], result_data['gluc_samples'][:10, mask].T, alpha=0.2, color=color)

def plot_quantiles(result_data, id, color):
    df = result_data['df_gluc']
    mask = result_data['df_gluc']['id']==id
    df = df[mask]
    plt.fill_between(df['time'], df['q25'], df['q75'], alpha=0.2, color=color)

def plot_line(data, id, label, color='b'):
    mask = data['df_gluc']['id']==id
    plt.plot(data['df_gluc']['time'][mask], data['df_gluc']['glucose'][mask], label=label, color=color)

def plot_id(data, id, label, color='b'):
    mask = data['df_gluc']['id']==id
    plt.scatter(data['df_gluc']['time'][mask], data['df_gluc']['glucose'][mask], label=label, color=color)

def plot_meal_timing(data, id, c='r', have_label=True):
    """[summary]

    Args:
        gen_data (dict): dictionary of generated dataset
        c (str, optional): color. Defaults to 'r'.
    """
    mask = data['id']==id
    time = data['time'][mask]
    nutrients = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']

    nutr = data[nutrients][mask].values
    #height = min(gen_data['glucose']) - 0.5
    y_bottom = np.zeros(nutr.shape[0])
    for i in range(nutr.shape[1]):
        label = nutrients[i]
        if (not have_label):
            label = None
        height_mult = 3
        plt.bar(time, height=height_mult*nutr[:,i], bottom=y_bottom, width=20, label=label)
        y_bottom = y_bottom + height_mult*nutr[:,i]

def plot_samples_grid(fit):
    """Creates a pairplot of posterior samples for one dimensional posterior parameters

    Args:
        fit CmdStanMCMC: the fitted model object
    """
    var_names = fit_varnames(fit)
    print(var_names)
    print(fit.draws_pd())
    az.plot_pair(
        fit.draws_pd(), divergences=True
    )

def plot_samples(fit):
    """Creates density plots for one dimensional posterior parameters 

    Args:
        fit CmdStanMCMC: the model object fit
    """
    var_names = fit_varnames(fit)
    print(fit.draws_pd()[var_names].to_dict())
    az.plot_density(
        fit.draws_pd()[var_names].to_dict(),
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

    meal_m = np.mean(fit.stan_variable('pred_meals_eiv'), axis=0)
    print(meal_m)
    #plt.vlines(gen_data['meal_timing'], 0, 1.2, label='observed meals', color='r')
    plot_meal_timing(gen_data)
    if 'true_timing' in gen_data:
        plt.vlines(gen_data['true_timing'], 0, 1.1, label='true timing', color='y')
    plt.vlines(meal_m, 0, 1, label='estimated meals', color='b')
    dy = 0
    dx = meal_m - gen_data['meal_timing']
    for i in range(dx.shape[0]):
        plt.arrow(gen_data['meal_timing'][i], 1, dx[i], dy, head_width=0.05, head_length=0.1, length_includes_head=True)
   

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

def plot_fit(fit, gen_data):
    """Plots shaded plot of glucose prediction
    and counter factual prediction (prediction if meal was not eaten)

    Args:
        fit CmdStanMCMC: the model object fit
    """
    y = fit.stan_variable('pred_gluc')
    x = gen_data['time']
    shadedplot(x, y, label='GP plot')
    y = fit.stan_variable('baseline_gp')
    shadedplot(x, y, label='CF', c='r')

def plot_baseline(fit, gen_data):
    """ plots the baseline and true value

    Args:
        fit CmdStanMCMC: the model object fit
        gen_data (dict): dictionary of generated dataset
    """
    if 'baseline' in gen_data:
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
    if 'resp' in gen_data:
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


    
    #plt.vlines(time, -1, height, label='observed meal_timing', color=c)

def shadedplot(x, y_samples, label, c='k', quantiles=[0.25, 0.75]):
    """[summary]

    Args:
        x (np.array(N)): N number of timesteps
        y_samples (np.array(S, N)): S samples, N timesteps
        label (str)
        c (str, optional): color. Defaults to 'k'.
    """
    median = np.quantile(y_samples, 0.5, axis=0)
    lower = np.quantile(y_samples, quantiles[0], axis=0)
    upper = np.quantile(y_samples, quantiles[1], axis=0)
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

