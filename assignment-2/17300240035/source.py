import os
os.sys.path.append('..')

from handout import Dataset, get_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
import PartI
import PartII
import numpy as np
from matplotlib import pyplot as plt


'''
#Part 1
d = get_linear_seperatable_2d_2c_dataset()

w = PartI.Least_Square_Model(d)
PartI.Display(d, w, 'Least Square Classification')
PartI.Accuracy(d, w, 'Accuracy of Least Square Classification')

w = PartI.Perceptron(d)
PartI.Display(d, w, 'Perceptron Algorithm')
PartI.Accuracy(d, w, 'Accuracy of Perceptron Algorithm')
'''

#Part 2
dataset_train, dataset_test = get_text_classification_datasets()
N = len(dataset_train.data)
vocab = PartII.Get_Vocabulary(dataset_train.data)
multi_hot, one_hot = PartII.Preprocess(dataset_train.data, dataset_train.target, len(dataset_train.target_names), vocab)
#Full-Batch-Gradient-Descent
#W, b, iter = PartII.Logistic_Regression(data = multi_hot, target = one_hot, epsilon = 1e-3, alpha = 0.25, batch = N)
#Mini-Batch-Gradient-Descent
#W, b, iter = PartII.Logistic_Regression(data = multi_hot, target = one_hot, epsilon = 1e-3, alpha = 0.25, batch = 128)
#Stochastic-Gradient-Descent
W, b, iter = PartII.Logistic_Regression(data = multi_hot, target = one_hot, epsilon = 1e-3, alpha = 0.01, batch = 1)
print("Accuracy on training set:", PartII.Classification_Accuracy(multi_hot, dataset_train.target, W, b))

'''
#Gradient_Checker
PartII.Gradient_Checker(data = multi_hot, target = one_hot)
'''

'''
#Select Learning Rate
xs = np.linspace(0.01, 0.6, 25)
ys = []
for Alpha in xs:
    W, b, iter = PartII.Logistic_Regression(data = multi_hot, target = one_hot, epsilon = 1e-3, lamb = 0.01, alpha = Alpha, batch = N, MAXITER = 40)
    ys.append(PartII.Classification_Accuracy(multi_hot, dataset_train.target, W, b))
    #ys.append(iter)
    print(Alpha, ys[len(ys) - 1])

plt.plot(xs, ys, color = "red")
plt.xlabel("Accuracy")
plt.ylabel("# of Iterations")
plt.show()
'''

multi_hot, one_hot = PartII.Preprocess(dataset_test.data, dataset_test.target, len(dataset_test.target_names), vocab)
print("Accuracy on test set:",PartII.Classification_Accuracy(multi_hot, dataset_test.target, W, b))

