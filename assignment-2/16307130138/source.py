import os
os.sys.path.append('../')
from handout import get_linear_seperatable_2d_2c_dataset as get2d2c
from handout import get_text_classification_datasets as get_text
import numpy as np
import matplotlib.pyplot as plt
import string
import time
import math
from Part1 import *
from Part2 import *


""" Part 1"""
# task = Part1()
# task.least_square_model()
# task.perceptron_model()

""" Part 2"""
# #general logistic model 
# task2 = Part2(epoch = 500)
# task2.logistic_model()

# #general logistic model - vary learning_rate
# task2 = Part2(epoch = 500)
# task2.test_learning_rate()

# #general logistic model - vary lambda 
# task2 = Part2(epoch = 500)
# task2.test_lamda()

# #SGD logistic model - set batch_size when initialize, default as full_bacth
# task2 = Part2(epoch = 2000,batch_size=500)
# task2.Logistic_SGD()

# #general logistic model - vary batch_size
# task2 = Part2(epoch = 500)
# task2.test_batch_size2()

# #check gradient
# task2 = Part2()
# if(task2.gradient_check(100)):
#     print("Gradient calculation is Correct")
