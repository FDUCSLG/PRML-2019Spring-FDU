import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import argparse
import numpy as np
import model

if __name__ == '__main__':
    # preprocess the input data and labels
    data_points = handout.get_linear_seperatable_2d_2c_dataset()
    target = [1 if i==True else -1 for i in data_points.y]
    new_input_data = np.array([np.append([1], i) for i in data_points.X])

    # build model and achieve the best weight
    least_square_model = model.Least_Square_model(new_input_data, target)
    weight = least_square_model.run()

    # plot the graph
    graph = plt.subplot(1,1,1)
    least_square_model.plot(graph)
    plt.legend(loc='best', prop={'size':20})
    #print(least_square_model.weight)
    plt.show()
