import M_0
import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
ans=[]
for i in range(1,20):
    ans.append(M_0.M_0(i*0.05))

plt.plot([i*0.5 for i in range(1,20)],ans)
plt.show()
