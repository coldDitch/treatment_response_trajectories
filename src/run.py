"""
    File for running the main experiment
"""

import matplotlib.pyplot as plt

from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response, plot_meal_pred, plot_samples_grid, plot_samples
from data_generators import generate_data, test_train_split
from model_utils import fit_model
from utils import summary

from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, NOISESCALE, SUMMARY, PLOTSAMPLES


gen_data = generate_data(days=DAYS, lengthscale=NOISESCALE)
train_data, test_data = test_train_split(gen_data, train_percentage=0.5)

fit = fit_model(train_data, test_data)

if DIAGNOSE:
    fit.diagnose()

if SUMMARY:
    summary(fit)

if PLOTFIT:
    plt.figure(figsize=FIGSIZE)
    plot_meal_pred(fit, gen_data)
    plot_datagen(train_data, test_data=test_data)
    plot_fit(fit)
    plt.legend()

if PLOTBASE:
    plt.figure(figsize=FIGSIZE)
    plot_baseline(fit, gen_data)
    plt.legend()

if PLOTRESP:
    plt.figure(figsize=FIGSIZE)
    plot_response(fit, gen_data)
    plt.legend()

if PLOTSAMPLES:
    plot_samples(fit)
    plot_samples_grid(fit)

plt.show()
