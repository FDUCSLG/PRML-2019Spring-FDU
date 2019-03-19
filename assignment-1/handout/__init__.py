from typing import List
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
np.random.seed(0)

class GaussianMixture1D:
    def __init__(self, num_mode=5, mode_range=(.0, 1.), std_range=(.0, 1.)):
        self.num_mode = num_mode
        self.mode_range = mode_range
        self.std_range = std_range
        self.modes = np.random.rand(num_mode) * (mode_range[1] - mode_range[0]) + mode_range[0]
        self.stds = np.random.rand(num_mode) * (std_range[1] - std_range[0]) + std_range[0]
        weights = np.random.rand(num_mode)
        self.weights = weights / weights.sum()
        # print(self.modes, self.stds, self.weights)

    def sample(self, shape):
        choices = np.random.choice(self.num_mode, shape, p=self.weights)
        modes = self.modes[choices]
        stds = self.stds[choices]
        sampled_data = np.random.normal(modes, stds)
        return sampled_data

    def plot(self, num_sample=100):
        sampled_data = self.sample([num_sample])
        min_range = min(self.modes) - 3 * self.std_range[1]
        max_range = max(self.modes) + 3 * self.std_range[1]
        xs = np.linspace(min_range, max_range, 2000)
        ys = np.zeros_like(xs)

        for l, s, w in zip(self.modes, self.stds, self.weights):
            ys += ss.norm.pdf(xs, loc=l, scale=s) * w

        plt.plot(xs, ys)
        # plt.hist(sampled_data, normed=True, bins=100)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()

gm1d = GaussianMixture1D(mode_range=(0, 50))
sampled_data = gm1d.sample([10000])

def get_data(num_data:int = 100) -> List[float]:
    """
    Please use this function to access the given distribution, you should provide an int
    `num_data` to indicate how many samples you want, note that num_data must be no
    larger than 10000
    """
    assert num_data <= 10000
    return list(sampled_data[:num_data])
# gm1d.plot(num_sample=1000)