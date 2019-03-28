# path to top module
import os
os.sys.path.append('..')

# lib
from handout import gm1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


def plot_gm1d():
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)

    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    plt.plot(xs, ys)
