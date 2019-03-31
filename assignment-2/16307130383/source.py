import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
from handout import get_linear_seperatable_2d_2c_dataset
import numpy as np
import matplotlib.pyplot as plt

def least_square_model():
  data = get_linear_seperatable_2d_2c_dataset()
  X = []
  for x in data.X:
    X.append([1, x[0], x[1]])
  X = np.array(X)
  # calculate the W matrix
  X_ = np.linalg.inv(X.transpose() @ X) @ X.transpose()
  T = np.array([int(x) for x in data.y])
  W = X_ @ T
  # draw the plot
  data.plot(plt)
  line_x = np.linspace(-2, 2, 10)
  line_y = (-W[1]*line_x + W[0]) / W[2]
  plt.plot(line_x, line_y)
  plt.title('y={} x {}'.format(-W[1]/W[2], W[0]/W[2]))
  plt.show()