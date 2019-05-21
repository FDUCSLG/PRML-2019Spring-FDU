import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import argparse
import numpy as np
import model

LEARNING_RATE = 0.02


if __name__ == '__main__':
    data_points = handout.get_linear_seperatable_2d_2c_dataset()
    target = [1 if i==True else -1 for i in data_points.y]
    new_input_data = np.array([np.append([1], i) for i in data_points.X])
    # build model
    perception_model = model.Perception_model(new_input_data, target, LEARNING_RATE)
    weight = perception_model.run()
    print(weight)
    graph = plt.subplot(1,1,1)
    perception_model.plot(graph)
    plt.legend(loc='best', prop={'size':14})
    plt.show()
         
    accuracy_val = (perception_model.accuracy_cal())
    print(accuracy_val)

