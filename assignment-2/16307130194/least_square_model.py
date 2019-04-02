import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_linear_seperatable_2d_2c_dataset


dataset = get_linear_seperatable_2d_2c_dataset()
dataset.plot(plt).show()
