import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
df = pd.read_csv('data/public_dataset.csv')
time = df['time'][df['id']==1].values.ravel()
y = gpr.sample_y(time)
plt.plot(time, y)
plt.show()