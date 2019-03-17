import M_NNM
import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
ans=[]
for i in range(2,50):
    ans.append(M_NNM.M_0(i))

plt.plot(range(2,50),ans)
plt.show()
