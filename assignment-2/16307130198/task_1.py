import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import argparse
import numpy as np
import model
import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["Least_Square_model", "Perception_model"], help="choose different kinds of models.")
parser.add_argument("--learning_rate", type=float, default=0.02, help="learning rate of the perception model")
parser.add_argument("--max_epoch", type=int, default=10, help="max epoch num")
args = parser.parse_args()

if __name__ == '__main__':
    # preprocess the input data and labels
    data_points = handout.get_linear_seperatable_2d_2c_dataset()
    target = [1 if i==True else -1 for i in data_points.y]
    new_input_data = np.array([np.append([1], i) for i in data_points.X])

    # build model and achieve the best weight
    temp = args.learning_rate
    """The Constructor of the Least Square model contains the **unused params, 
        so the form of calling the constructor can be unified."""
    classification_model = utils.find_class_by_name(args.model, [model])(new_input_data, target, learning_rate=args.learning_rate, max_epoch=args.max_epoch)
    weight = classification_model.run() 

    # plot the graph
    graph = plt.subplot(1,1,1)
    classification_model.plot(graph)
    plt.legend(loc='best', prop={'size':14})
    plt.show()
