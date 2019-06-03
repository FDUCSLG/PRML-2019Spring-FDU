import os
import sys
from pprint import pprint 
os.sys.path.append('..')
from handout import *
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import string



data_set = get_linear_seperatable_2d_2c_dataset()

class BASE:

	def __init__(self, dataset, results, plot_title ):
		self.plot_title = plot_title
		self.dataset = data_set
		self.w = results 

	# Generates the accuracy of the results
	def generate_accuracy(self, d):  # accuracy over test set
		pred_y = [self.classify(x) for x in d.X]
		return d.acc(pred_y)

	def plot_graph(self, plt, title):
		# Setup figure
		fig = plt.figure()
		plt = self.dataset.plot(plt)
		plt = self.draw_line(plt)
		plt.title(self.plot_title+", Acc=%f" % result.generate_accuracy(data_set))
		plt.show()


class Perceptron(BASE):
	def __init__(self, data_set):
		super().__init__(data_set,self.algoritm(data_set), "Perceptron plot")

	# the perception algoritm 
	def algoritm(self, d):
		X = np.array(d.X)
		b = np.ones(len(d.X))
		X = np.mat(np.insert(X, 0, values=b, axis=1))

		# initial weight
		w = np.random.rand(3, 1)

		# compute t
		t = [1 if yy else -1 for yy in d.y]

		for i in range(len(X)):
			if np.dot(X[i], w)*t[i] < 0:  # misclassification
				w = w + X[i].T*t[i]
		return w

	# Helper method to classify 
	def classify(self, X):
		X = np.mat([1, X[0], X[1]]).reshape(3, 1)
		result = np.dot(self.w.T, X)
		if result[0] < 0:
			return False
		else:
			return True

	# Draw a decision line on top of the dataset to visually 
	# show to separate the dataset.
	def draw_line(self, plt):
		w = self.w.tolist()
		k = -w[1][0]/w[2][0]
		b = -w[0][0]/w[2][0]
		plt.plot([-1.5, 1.0], [-1.5 * k + b, 1.0 * k + b])
		return plt


class LeastSquare(BASE):
	def __init__(self, data_set):
		super().__init__(data_set,self.algoritm(data_set), "Least Square plot")
		
	# the Least Square algoritm 
	def algoritm(self, d):
		# compute X~
		X = np.array(d.X)
		b = np.ones(len(d.X))
		X = np.mat(np.insert(X, 0, values=b, axis=1))
		X_trans = X.T
		X_dot = np.dot(X_trans, X)
		X_pseudo_inverse = np.dot(np.mat(X_dot).I, X_trans)
		# turn T into one-hot representation
		T = np.array([[0, 1] if yy else [1, 0] for yy in d.y])

		w= np.dot(X_pseudo_inverse, T)
		return w

	def classify(self, X):  # classify a sample point
		X = np.mat([1, X[0], X[1]]).reshape(3, 1)
		result = np.dot(self.w.T, X)
		if result[0] > result[1]:
			return False
		else:
			return True

	def draw_line(self, plt):
		W = self.w.tolist()
		k = (W[1][0]-W[1][1])/(W[2][1]-W[2][0])
		b = (W[0][0]-W[0][1])/(W[2][1]-W[2][0])
		plt.plot([-1.5, 1.0], [-1.5 * k + b, 1.0 * k + b])
		return plt



# Part 2


# Softmax for a matrix
def Softmax(z):
	if z.ndim > 1:
		return [softmax(zz) for zz in z]
	return softmax(z)

# softmax with offset
def softmax(z):
	z = z - np.max(z)
	return np.exp(z)/np.sum(np.exp(z))


def combine_whitespace(s):
	return s.split()

class LogisticTextClassification:
	def __init__(self):
		self.min_count = 10
		self.learning_rate =1
		self.steps = 500
		self.step_array = [i for i in range(self.steps)]
		self.text_train, self.text_test = get_text_classification_datasets()
		self.train_vec = []
		self.test_vec = []
		self.vocab = []
		self.categories = self.text_train.target_names
		self.X, self.target_vec = self.preprocessing()
		
		# Setup initial params
		W = np.random.rand(4, len(self.X[0].tolist())) 	# Generate random weights 
		N = len(self.X) 	# Size
		B=5	# Batch size
		loss = [self.cross_entropy_loss(W)] # Initial loss
		

		#Modify here to use the three different ways
		self.W = self.logistic(W,N,B,loss)
		#self.W = self.logistic_batched(W,N,B,loss)
		#self.W = self.logistic_stochastic(W,N,B,loss)

	def tokenize_data(self, data):
		for x,line in enumerate(data):
			line = line.lower()
			for c in string.punctuation:
				line = line.replace(c, "")
			for w in string.whitespace:
				line = line.replace(w, " ")
			words = line.split()
			data[x]=words
			
		word_dict = {}
		for text in data:
			for word in text:
				if word in word_dict.keys():
					word_dict[word] += 1
				else:
					word_dict[word] = 1
		
		word_list = list(word_dict.keys())
		for word in word_list:
			if word_dict[word] < self.min_count:
				del word_dict[word]

		for x,line in enumerate(data):
			for word in line:
				if word not in word_dict.keys():
					data[x].remove(word);

		return data

	# vector representation of input and target
	def preprocessing(self):
		#self.tokenize()
		# Cleaning up the words
		self.text_train.data=self.tokenize_data(self.text_train.data)
		self.text_test.data=self.tokenize_data(self.text_test.data)

		self.train_vec=self.text_train.data
		# Format the vocab
		self.vocab = sorted(list(set([w for line in self.train_vec for w in line])))
		self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
		# One dim hot
		target_vec = np.zeros((len(self.text_train.target), 4))
		for i in range(len(target_vec)):
			target_vec[i][self.text_train.target[i]] = 1.0
		# Multi dim hot
		multi_hot_vec = np.zeros((len(self.train_vec), len(self.vocab)))
		for i in range(len(self.train_vec)):
			for j in range(len(self.train_vec[i])):
				multi_hot_vec[i][self.vocab_to_idx[self.train_vec[i][j]]] = 1.0
		X = multi_hot_vec
		b = np.ones(len(X))
		X = np.insert(X, 0, values=b, axis=1)
		return X, target_vec

	def compute_partial(self, W):  # W:(1+D)*4, x:n*(1+D) n training samples，1+D dimensions
		error = Softmax(np.dot(self.X, W.T)) - self.target_vec
		N = len(self.X)
		return 1/N * np.dot(error.T, self.X)

	def compute_batched_partial(self, W, start, end):
		error = Softmax(np.dot(self.X[start:end], W.T)) - self.target_vec[start:end]
		N = end - start
		return 1/N * np.dot(error.T, self.X[start:end])


	def cross_entropy_loss(self, W):
		N = len(self.X)
		return -1/N * sum(math.log(Softmax(np.dot(self.X[i], W.T))[self.text_train.target[i]]) for i in range(N))


	def logistic(self,W,N,B,loss):  # full batch 0.0082
		for i in range(self.steps):
			W = W - self.learning_rate * (max(0.95 ** i, 0.1)) * self.compute_partial(W)
			new_loss = self.cross_entropy_loss(W)
			print("loss,", new_loss, "steps ", i+1)

			loss.append(new_loss)

		plt.plot(self.step_array[:min(len(loss), self.steps)], loss[:min(len(loss), self.steps)])
		plt.show()
		return W

	def logistic_stochastic(self,W,N,B,loss):  # stochastic gradient descent
		for i in range(self.steps):
			n = np.random.randint(0, N-1)  # select a sample randomly to update W
			W = W - self.learning_rate * (max(0.95 ** i, 0.1)) * self.compute_batched_partial(W, n, n+1)
			new_loss = self.cross_entropy_loss(W)
			loss.append(new_loss)
			print(loss)

		plt.plot(self.step_array[:min(len(loss), self.steps)], loss[:min(len(loss), self.steps)])
		plt.show()
		return W

	def logistic_batched(self,W,N,B,loss):  # batched gradient descent
		for i in range(0, N, B):
			W = W - self.learning_rate  * (max(0.95 ** i, 0.1)) * self.compute_batched_partial(W, i, i + B)
			new_loss = self.cross_entropy_loss(W)
			loss.append(new_loss)
			print(new_loss,i)

		plt.plot(self.step_array[:min(len(loss), self.steps)], loss[:min(len(loss), self.steps)])
		plt.show()
		return W

	def classify(self, W, xn):
		prob = Softmax(np.dot(xn, W.T))
		return prob.tolist().index(max(prob))

	def compute_accuracy(self):
		self.test_vec = self.text_test.data
		multi_hot_vec = np.zeros((len(self.test_vec), len(self.vocab)))
		for i in range(len(self.test_vec)):
			for j in range(len(self.test_vec[i])):
				print("compute_accuracy debug", i,len(self.test_vec),j,len(self.test_vec[i]) )
				if self.test_vec[i][j] in self.vocab: 
					multi_hot_vec[i][self.vocab_to_idx[self.test_vec[i][j]]] = 1.0

		test_target_vec = np.zeros((len(self.text_test.target), 4))
		for i in range(len(test_target_vec)):
			test_target_vec[i][self.text_test.target[i]] = 1.0

		test_X = multi_hot_vec
		b = np.ones(len(test_X))
		test_X = np.insert(test_X, 0, values=b, axis=1)
		right = 0

		for i in range(len(test_X)):
			right = right + (self.classify(self.W, test_X[i]) == self.text_test.target[i])

		return right/len(self.text_test.data)


if __name__ == '__main__':  
	# Part 1
	#result = Perceptron(data_set)
	#result.plot_graph(plt,"Perceptron plot")
	#print("Accuracy from Perception", result.generate_accuracy(data_set))

	#result = LeastSquare(data_set)
	#result.plot_graph(plt,"Least Square plot")
	#print("Accuracy from Least Square", result.generate_accuracy(data_set))

	# Part 2
	ML = LogisticTextClassification()
	print("cross_entropy_loss")
	print(ML.cross_entropy_loss(ML.W))
	print("compute_accuracy")
	print(ML.compute_accuracy())

