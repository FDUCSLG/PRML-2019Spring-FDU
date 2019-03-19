import os
os.sys.path.append('..')
from handout import get_data
from handout import GaussianMixture1D
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

sampled_data = get_data(100)
xmin = min(sampled_data)
xmax = max(sampled_data)
x = np.linspace(xmin, xmax, 100)

def realplot(num_sample = 1000):
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    self = gm1d
    sampled_data = self.sample([num_sample])
    min_range = min(self.modes) - 3 * self.std_range[1]
    max_range = max(self.modes) + 3 * self.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(self.modes, self.stds, self.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)

def CV(h):
    KL = 0
    kf = KFold(n_splits = 10, shuffle = False)
    for train_index, test_index in kf.split(sampled_data):
        #print('Train index:', train_index)
        #print('test index:', test_index)
        train_Set = []
        validation_Set = []
        for i in train_index:
            train_Set.append(sampled_data[i])
        for j in test_index:
            validation_Set.append(sampled_data[j])
        KL += Gausset(train_Set, validation_Set, h)
    KL /= 10
    return KL

def Gausset(train_Set, validation_Set, h):
    sc = []
    for v in validation_Set:
        tmp = 0
        for t in train_Set:
            tmp += 1 / (2 * math.pi * h**2)**(1 / 2) * math.exp(-(v - t)**2 / (2*h**2))
        sc.append(math.log(tmp / 90))
    return sum(sc)

score = []
h = np.linspace(0.1, 0.6, 50)
for hh in h:
    score.append(CV(hh))
   #print("h: ", hh,"CV: ", CV(hh))

plt.plot(h, score)
plt.title('KL divergence - h')
best = h[np.argmax(score)]
print("best h:", best)

Gauss = np.zeros(shape = 100)
for i in range(len(x)):
    for j in range(len(sampled_data)):
        Gauss[i] += np.exp(-(x[i] - sampled_data[j])**2 / (2 * best**2)) / (best**2 * 2 * math.pi)**(1 / 2)
    Gauss[i] /= 100

#plt.hist(sampled_data, normed=True, bins=50)
#plt.plot(x, Gauss, color = 'blue')
#realplot()
plt.show()

