"""
    File for running the main experiment
"""

import matplotlib.pyplot as plt
from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response, plot_meal_pred, plot_samples_grid, plot_samples
from data_generators import generate_data, test_train_split
from model_utils import fit_model
from utils import summary
from preprocess import public_data
from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, SUMMARY, PLOTSAMPLES, SYNTHETIC
from metrics import print_metrics


def main():
    if SYNTHETIC:
        data = generate_data(days=DAYS)
    else:
        data = public_data(id=6)
    train_data, test_data = test_train_split(data, train_percentage=0.33)
    print('num observations', test_data['time'].shape)
    print('train observations', train_data['time'].shape)
    fit = fit_model(train_data, test_data)
    print_metrics(fit, train_data, test_data)
    if DIAGNOSE:
        fit.diagnose()
    if SUMMARY:
        summary(fit)
    if PLOTFIT:
        plt.figure(figsize=FIGSIZE)
        plot_meal_pred(fit, data)
        plot_datagen(train_data, test_data=test_data)
        plot_fit(fit, data)
        plt.legend()
    if PLOTBASE:
        plt.figure(figsize=FIGSIZE)
        plot_baseline(fit, data)
        plt.legend()
    if PLOTRESP:
        plt.figure(figsize=FIGSIZE)
        plot_response(fit, data)
        plt.legend()
    if PLOTSAMPLES:
        plot_samples(fit)
        plot_samples_grid(fit)
    plt.show()

if __name__ == '__main__':
    main()