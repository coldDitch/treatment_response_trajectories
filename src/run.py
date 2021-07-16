import matplotlib.pyplot as plt

from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response
from data_generators import generate_data, test_train_split
from model_utils import fit_model

from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, NOISESCALE, SUMMARY

gen_data = generate_data(days=DAYS, lengthscale=NOISESCALE)
train_data, test_data = test_train_split(gen_data)

fit = fit_model(train_data, test_data)

if DIAGNOSE:
    fit.diagnose()

if SUMMARY:
    means = fit.summary()['Mean']
    for key, val in means.iteritems():
        if not '[' in key:
            print(key, val)

if PLOTFIT:
    plt.figure(figsize=FIGSIZE)
    plot_fit(fit)
    plot_datagen(train_data, test_data=test_data, gen_data=gen_data)
    plt.legend()
    plt.show()

if PLOTBASE:
    plt.figure(figsize=FIGSIZE)
    plot_baseline(fit, gen_data)
    plt.legend()
    plt.show()

if PLOTRESP:
    plt.figure(figsize=FIGSIZE)
    plot_response(fit, gen_data)
    plt.legend()
    plt.show()