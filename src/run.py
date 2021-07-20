import matplotlib.pyplot as plt

from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response, plot_meal_pred
from data_generators import generate_data, test_train_split
from model_utils import fit_model
from utils import summary

from config import DIAGNOSE, PLOTFIT, PLOTBASE, PLOTRESP, FIGSIZE, DAYS, NOISESCALE, SUMMARY

gen_data = generate_data(days=DAYS, lengthscale=NOISESCALE)
train_data, test_data = test_train_split(gen_data)


fit = fit_model(train_data, test_data)


plot_meal_pred(train_data, fit)
plt.legend()

if DIAGNOSE:
    fit.diagnose()

if SUMMARY:
    summary(fit)

if PLOTFIT:
    plt.figure(figsize=FIGSIZE)
    plot_datagen(train_data, test_data=test_data, gen_data=gen_data)
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


plt.show()