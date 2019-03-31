import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import gen_linear_seperatable_2d_2c_dataset
import numpy as np
import matplotlib.pyplot as plt
import math

d = get_linear_seperatable_2d_2c_dataset()
d.plot(plt).show()