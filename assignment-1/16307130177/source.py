import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt


sampled_data = get_data(500)
plt.hist(sampled_data, normed=True, bins=50)
plt.show()
