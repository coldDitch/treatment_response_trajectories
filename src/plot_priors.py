import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv
from scipy.stats import geninvgauss


x = np.linspace(0, 2, 100)
p, a, b = 1, 5, 1

def geninvstan(x, p, a, b):
    lpdf = p * 0.5 * np.log(a / b) - np.log(2 * kv(p, np.sqrt(a * b))) + (p - 1) * np.log(x) - (a * x + b / x) * 0.5
    return np.exp(lpdf)

def inv_gaussian(x, mu, l):
    pdf = np.sqrt(l/(2*np.pi * x **3)) * np.exp(-l*(x-mu)**2/(2*mu**2*x))
    return pdf

def inv_l_gaussian(x,mu, l):
   lpdf = 0.5 * (-l  * (mu - x)**2/(mu**2 * x) + np.log(l) - 3 * np.log(x) - np.log(2 * np.pi))
   return np.exp(lpdf)

y = inv_gaussian(x, 3, 0.1)
plt.plot(x,y, c='b')

y = inv_l_gaussian(x, 0.5, 0.3)
plt.plot(x,y, c='r')
plt.show()