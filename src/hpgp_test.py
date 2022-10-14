import cmdstanpy
import matplotlib.pyplot as plt
from data_generators import test_train_split
from model_utils import combine_to_prediction_data
from preprocess import public_data
from plot_utils import plot_datagen, shadedplot


id = 6
data = public_data(id)

train, test = test_train_split(data, train_percentage=0.8)
pred_data = combine_to_prediction_data(train, test)
dat = train.copy()
dat.update(pred_data)
dat['L'] = data['time'].max() * 5/2
dat['M'] = 120
print(dat.keys())

bayesname = 'HBGP'
bayespath = 'stan_models/' + bayesname + '.stan'
options = {"STAN_THREADS": True}
model = cmdstanpy.CmdStanModel(stan_file=bayespath, cpp_options=options)
fit = model.sample(data=dat,
    output_dir='logs',
    parallel_chains=4,
    threads_per_chain=2,
    show_progress=True,
    seed=1234,
    iter_warmup=1000)


x = data['time']
y = fit.stan_variable('pred_gluc')

print('shape')
print(x.shape)
print(y.shape)
shadedplot(x,y, label='test')
plot_datagen(train, test)
plt.show()