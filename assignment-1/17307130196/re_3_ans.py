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
import Nearest_Neighbor_Method
# sample_data=get_data()
num=1000
esti=Kernel_Method.Kernel(1000,100,0.4)
# esti_nnm=Nearest_Neighbor_Method.KNN(1000,100,15)
# plt.plot(np.linspace(0,50,num),esti_nnm)
# esti=Histogram_Method.Hist(100,20)
plt.plot(np.linspace(0,50,num),esti)
# gm1d = GaussianMixture1D(mode_range=(20, 40))
# gm1d.plot(100)
plt.show()