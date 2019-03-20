import os
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

sampled_data = get_data(1000)
plt.hist(sampled_data, normed=True, bins=2,color='blue',label='bins=2',alpha=0.4)

sampled_data = get_data(1000)
plt.hist(sampled_data, normed=True, bins=10,color='orange', label='bins=10',alpha=0.4)

sampled_data = get_data(1000)
plt.hist(sampled_data, normed=True, bins=100,color='red',label='bins=100',alpha=0.4)

sampled_data = get_data(1000)
plt.hist(sampled_data, normed=True, bins=500,color='grey',label='bins=100',alpha=0.4)

plt.legend()
plt.title('Figure 2-5')
plt.show()
