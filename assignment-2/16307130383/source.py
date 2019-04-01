import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
from handout import get_linear_seperatable_2d_2c_dataset
import numpy as np
import matplotlib.pyplot as plt

#------------
# Part 1
#------------
def least_square_model():
  data = get_linear_seperatable_2d_2c_dataset()
  X = []
  for x in data.X:
    X.append([1, x[0], x[1]])
  X = np.array(X)
  # calculate the W matrix
  X_ = np.linalg.inv(X.transpose() @ X) @ X.transpose()
  T = np.array([[int(x), 1-int(x)] for x in data.y])
  W = X_ @ T
  # draw the plot
  data.plot(plt)
  line_x = np.linspace(-2, 2, 10)
  line_y = -( W[0][0]-W[0][1]+( W[1][0]-W[1][1] )*line_x )/( W[2][0]-W[2][1] )
  plt.plot(line_x, line_y)
  plt.title('y={} x {}'.format(-( W[1][0]-W[1][1] )/( W[2][0]-W[2][1] ), -( W[0][0]-W[0][1] )/( W[2][0]-W[2][1] )))
  plt.show()

def perceptron_algorithm():
  data = get_linear_seperatable_2d_2c_dataset()
  w = np.array([-1.0, 1.0])
  offset = np.array([0.1, -0.30])
  # within 3 cycles the w can be figure out
  for cycle in range(3):
    for x, y in zip(data.X, data.y):
      x_ = x - offset
      if (x_ @ w) * (int(y) - 0.5) < 0:
        w += x_ * (int(y) - 0.5) * 2
        data.plot(plt)
        plt.scatter([x[0]], [x[1]], c='#ff0000')
        plt.scatter([offset[0]],[offset[1]], c='#0000ff')
        line_x = np.linspace(-1.7, 1.7, 10)
        line_y = -w[0]/w[1] * (line_x - offset[0]) + offset[1]
        plt.plot(line_x, line_y)
        plt.show()