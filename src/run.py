import matplotlib.pyplot as plt

from plot_utils import plot_datagen, plot_fit
from data_generators import generate_data, test_train_split
from model_utils import fit_model

gen_data = generate_data()
train_data, test_data = test_train_split(gen_data)

fit = fit_model(train_data)

plt.figure(figsize=(16,9))
plot_fit(fit)
plot_datagen(gen_data)
plt.legend()
plt.show()