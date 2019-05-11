import os
os.sys.path.append('..')
import argparse
import numpy as np

from least_square_model import LSM
from perceptron import Perceptron
from logistic import Logistic

from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets


def program_parser():
    parser = argparse.ArgumentParser(description='Assignment 2')

    parser.add_argument('--algorithm',
                        choices=["least_square", "perceptron", "logistic"],
                        help='the algorithms')

    parser.add_argument('--n',
                        choices=["run", "batch", "lambda", "alpha", "check"],
                        default="run",
                        help='the algorithms of logistic')

    args = parser.parse_args()

    linear_dataset = get_linear_seperatable_2d_2c_dataset()
    lsm = LSM(linear_dataset)
    perceptron = Perceptron(linear_dataset)

    algos = {
        "least_square": lsm.run,
        "perceptron": perceptron.run
    }

    if args.algorithm == "logistic":
        np.random.seed(2333)
        dataset_train, dataset_test = get_text_classification_datasets()
        logistic = Logistic(dataset_train, dataset_test)
        if args.n == "run":
            logistic.show()
        elif args.n == "check":
            logistic.check_gradient()
        elif args.n == "batch":
            logistic.show_batch_diff()
        elif args.n == "lambda":
            logistic.show_lamb_diff()
        elif args.n == "alpha":
            logistic.show_alpha_diff()
    elif args.algorithm in algos.keys():
        algos[args.algorithm]()
    else:
        parser.print_help()


if __name__ == "__main__":
    program_parser()
