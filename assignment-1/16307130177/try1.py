import os
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold

sampled_data = get_data(100)
xmin = min(sampled_data)
xmax = max(sampled_data)
x = np.linspace(xmin, xmax, 100)

#def CV(h):
   # KL = 0
kf = KFold(n_splits = 10, shuffle = False)
for train_index, test_index in kf.split(sampled_data): 
    print('Train index:', train_index)
    print('test index:', test_index)
    train_Set = []
    validation_Set = []
    for i in train_index:
        train_Set.append(sampled_data[i])
    for j in test_index:
     validation_Set.append(sampled_data[j])
        #KL += Gausset(train_Set, validation_Set, h)
    #KL /= 10

