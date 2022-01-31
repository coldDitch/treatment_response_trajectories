import cmdstanpy
import matplotlib.pyplot as plt
import numpy as np
from preprocess import public_data
from data_generators import update_n
from plot_utils import shadedplot

SEED = 1234
dat = public_data()
dat['time'] = np.linspace(0, 5, 200)
update_n(dat)
model = cmdstanpy.CmdStanModel(stan_file='tests/prior_sample.stan')
fit = model.sample(data=dat, chains=1, seed=SEED, show_progress=True, output_dir='logs')
Y = fit.stan_variable('impulses')
x = dat['time']
print(Y[:,0,:].shape)
print(x.shape)
for meal in range(Y.shape[1]):
    plt.figure()
    shadedplot(x, Y[:, meal, :], label='impulse', quantiles=[0.1, 0.9])
    plt.show()

