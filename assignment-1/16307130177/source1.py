import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import iqr

sampled_data = get_data(100)
xmin = min(sampled_data)
xmax = max(sampled_data)
x = np.linspace(xmin, xmax, 100)
sigma = 1.06 * pow(100, -1/5) * min(np.std(sampled_data, ddof = 0), iqr(sampled_data) / 1.34)

Gaussest = np.zeros(shape = 100)
for i in range(len(x)):
    for j in range(len(sampled_data)):
        Gaussest[i] += np.exp(-(x[i] - sampled_data[j])**2 / (2 * sigma**2)) / (sigma**2 * 2 * math.pi)**(1 / 2)
    Gaussest[i] /= 100

plt.hist(sampled_data, normed=True, bins=50)
plt.plot(x, Gaussest, color = 'blue')
plt.show()

