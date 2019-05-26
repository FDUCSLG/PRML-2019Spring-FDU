import _pickle as pickle
import sys
import os
import torch
import models
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import tqdm

from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler 

import numpy as np


input_size = 30
h_size = 10
learning_rate = 0.001
vocab_size = 8

def main():
    # randomly generate input data
    input_data = np.random.rand(30, 1)
    targets = [i for i in range(vocab_size)]

    model = models.npLSTM(input_size=input_size, H_size=h_size, learning_rate=learning_rate)
    
    h_prev = np.ones((h_size, 1))
    C_prev = np.ones((h_size, 1))

    loss_val, h_val, C_val = model.forward_backward(
        input_data, targets, h_prev, C_prev 
    )
    
    # gain all the parameters
    variable_list = model.all()
    
    # gradient check
    Variable_list_copy = np.copy(variable_list)
    loss_copy = loss_val  

    # numerical cal the gradients
    step = 0
    dif_sum = 0
    dif_list = []
    delta = 1e-8
    
    for variable in variable_list:
        v_first, v_second = variable.v.shape
        for i in range(v_first):
            for j in range(v_second):
                old_variable_gradient = variable.d[i][j] 
                
                h_prev = np.ones((h_size, 1))
                C_prev = np.ones((h_size, 1))
                
                loss_modify, _, _ = model.forward_backward(
                        input_data, targets, h_prev, C_prev)
                
                variable.v[i][j] += delta
                loss_modify, _, _ = model.forward_backward(
                        input_data, targets, h_prev, C_prev)
                                
                variable.v[i][j] -= delta 
                temp = np.abs((loss_modify-loss_val)/delta - old_variable_gradient)
                dif_list.append(temp) 
                dif_sum += np.abs((loss_modify-loss_val)/delta - old_variable_gradient)
                step+= 1
    
    
    print("DIFFERENT LIST")
    print(str(dif_list[0:8])[0:-1]+"   ................   "+str(dif_list[-8:])[1:])
    print("DIFFERENT MEAN: \n" + str(dif_sum/step))




if __name__ == '__main__':
    main()




