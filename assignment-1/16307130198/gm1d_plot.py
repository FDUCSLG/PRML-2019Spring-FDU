import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as ss

np.random.seed(0)

def gm1d_plot(gm1d, graph=plt):
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 200)
    ys = np.zeros_like(xs)

    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    graph.plot(xs, ys, label="Real Line")
