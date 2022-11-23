import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


df = pd.read_csv('data/public_dataset.csv')
df = df[df['glucose'].notna()]
id = 2
df_person = df[df['id']==id]
print(df_person.head())
x = (df_person['time'].values).reshape(-1,1)
y = df_person['glucose'].values.ravel()
kernel = RBF(length_scale=20)
gpr = GaussianProcessRegressor(kernel=kernel,normalize_y=False)
g = gpr.sample_y(x, random_state=1234).ravel() 
print(g.std())

plt.figure(figsize=(50, 5))
plt.grid(markevery=1)
plt.plot(x, g, label='noise')
plt.plot(x, y, label='original glucose')
plt.plot(x, y+g, label='noise added')
plt.xlabel('time (minutes)')
plt.ylabel('glucose (mol/l)')
plt.show()