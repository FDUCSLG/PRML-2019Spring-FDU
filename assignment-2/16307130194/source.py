import os
os.sys.path.append('..')
import argparse

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

    args = parser.parse_args()

    linear_dataset = get_linear_seperatable_2d_2c_dataset()
    lsm = LSM(linear_dataset)
    perceptron = Perceptron(linear_dataset)

    algos = {
        "least_square": lsm.run,
        "perceptron": perceptron.run
    }

    if args.algorithm == "logistic":
        dataset_train, dataset_test = get_text_classification_datasets()
        logistic = Logistic(dataset_train, dataset_test)
        logistic.run()
    elif args.algorithm in algos.keys():
        algos[args.algorithm]()
    else:
        parser.print_help()


if __name__ == "__main__":
    program_parser()
