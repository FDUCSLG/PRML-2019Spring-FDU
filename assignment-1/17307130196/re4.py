import Nearest_Neighbor_Method
import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
from handout import GaussianMixture1D
import numpy as np
import matplotlib.pyplot as plt
import Kernel_Method


Nearest_Neighbor_Method.KNN(1000,500,3)
Nearest_Neighbor_Method.KNN(1000,500,6)
Nearest_Neighbor_Method.KNN(1000,500,9)
Nearest_Neighbor_Method.KNN(1000,500,12)
Nearest_Neighbor_Method.KNN(1000,500,24)
Nearest_Neighbor_Method.KNN(1000,500,36)

gm1d = GaussianMixture1D(mode_range=(0, 50))
sampled_data = gm1d.sample([10000])
Kernel_Method.Kernel(1000,200,0.4)
# gm1d.plot(200)

plt.show()