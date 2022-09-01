import matplotlib.pyplot as plt
from preprocess import public_data
from plot_utils import plot_baseline, plot_datagen, plot_fit, plot_response, plot_meal_pred, plot_samples_grid, plot_samples, plot_meal_timing

for i in range(14):
    data = public_data(id=i)
    plt.figure(figsize=(16,4))
    plt.title(str(i))
    plot_meal_timing(data)
    plot_datagen(data)
    plt.show()