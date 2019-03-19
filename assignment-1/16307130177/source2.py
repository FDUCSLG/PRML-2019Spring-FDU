import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

def Euclid(x, data, num = 5):
    distance = np.zeros(shape = len(data))
    for i in range(len(data)):
        distance[i] = abs(data[i] - x)
    distance.sort()
    return num / (distance[num] * len(data))

KNN = np.zeros(shape = 100)
K = 1
sampled_data = get_data(100)
xmin = min(sampled_data)
xmax = max(sampled_data)
x = np.linspace(xmin, xmax, 100)

for i in range(len(x)): 
    KNN[i] = Euclid(x[i], sampled_data, K) 
plt.plot(x, KNN)
#plt.hist(sampled_data, normed=True, bins=50)
plt.show()
