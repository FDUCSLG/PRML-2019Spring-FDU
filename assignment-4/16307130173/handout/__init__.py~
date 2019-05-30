import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def acc(self, pred_y):
        return float((self.y == pred_y).mean())

    def plot(self, plt):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        return plt

    def split_dataset(self, ratio=0.9):
        N = self.X.shape[0]
        idx = np.random.rand(N) < ratio
        ds_a = Dataset(self.X[idx], self.y[idx])
        ds_b = Dataset(self.X[~idx], self.y[~idx])
        return ds_a, ds_b


def get_linear_seperatable_2d_2c_dataset(ori=np.array([[0.1, -0.4]]),
                                         vec=np.array([[0.32, 0.62]]), num_points=200, ratio=0.5, gap=0.1):
    rng = np.random.RandomState(0)
    ort = np.array([[vec[0][1], -vec[0][0]]])
    y = rng.rand(num_points, 1) > ratio
    c = y * 2 - 1
    t = rng.normal(size=[num_points, 1])
    base = ori + t @ vec
    offset = ((rng.rand(num_points, 1) + gap) * c) @ ort
    X = base + offset
    return Dataset(X, y.reshape(-1))


def get_text_classification_datasets():
  #  categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
   # dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
    dataset_train = fetch_20newsgroups(subset='train')
    dataset_test = fetch_20newsgroups(subset='test')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test

if __name__ == "__main__":
    d = get_linear_seperatable_2d_2c_dataset()
    d.plot(plt).show()
