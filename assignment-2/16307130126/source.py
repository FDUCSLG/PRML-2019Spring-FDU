import os
import string
import time

os.sys.path.append('..')
import handout
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def get_accuracy(W, X, Y):
	accuracy = 0
	N = len(X)
	for i in range(N):
		if np.dot(W.T, X[i]) * Y[i] > 0:
			accuracy += 1
	return accuracy / N


def least_square(X, Y):
	W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
	return W


def perceptron(X, Y):
	N = len(X)
	times = 10000
	W = np.random.randn(3)
	rate = 0.1
	for i in range(times):
		p = i % N
		if p == 0:
			print(get_accuracy(W, X, Y))
		if np.dot(W, X[p]) * Y[p] < 0:
			W += rate * X[p] * Y[p]
	return W


def task1():
	dataset = handout.get_linear_seperatable_2d_2c_dataset()
	N = len(dataset.X)
	# print(len(dataset.X))

	w0 = np.ones([len(dataset.X), 1])
	# print(w0)
	X = np.concatenate((w0, dataset.X), axis=1)
	Y = np.zeros(N)
	for i in range(N):
		if dataset.y[i]:
			Y[i] = 1
		else:
			Y[i] = -1
	# print(X)
	W = None
	method = input('Least square model or perceptron algorithm? (l/p)')
	if method == 'l':
		W = least_square(X, Y)
	elif method == 'p':
		W = perceptron(X, Y)

	print("W = ", W)
	print("accuracy rate = ", get_accuracy(W, X, Y))

	min_data = min(X.T[1])
	max_data = max(X.T[1])
	x = np.arange(min_data, max_data, 0.01)
	y = (W[0] + W[1] * x) / (-W[2])

	plt.plot(x, y)
	plt.scatter(X[:, 1], X[:, 2], c=dataset.y)
	plt.plot(x, y)
	plt.show()


def deal_text(data):
	ret = []
	for text in data:
		s = ''
		for i in range(len(text)):
			if text[i] not in string.punctuation:
				if text[i] in string.whitespace:
					s += ' '
				elif ord('A') <= ord(text[i]) <= ord('Z'):
					s += chr(ord(text[i]) - ord('A') + ord('a'))
				else:
					s += text[i]
		ret.append(s)
	return ret


def deal_dateset(dataset, vocabulary):
	vocabulary_num = len(vocabulary)
	global categories_size
	N = len(dataset.data)
	print(N)
	X = np.zeros((N, vocabulary_num))
	Y = np.zeros((N, categories_size))
	for i in range(N):
		text = dataset.data[i]
		words = text.split(' ')
		for word in words:
			if word in vocabulary:
				X[i][vocabulary[word]] = 1
		Y[i][dataset.target[i]] = 1
	return X, Y


def check_gradient(X, Y, W, b, deltaW, deltaB):
	print("check gradient")
	epsilon = 0.001
	global categories_size, dimension
	loss, accuracy = test(X, Y, W, b)
	for i in range(dimension):
		for j in range(categories_size):
			tmpW = W.copy()
			tmpW[i][j] = tmpW[i][j] + epsilon
			c_loss, c_accuracy = test(X, Y, tmpW, b)
			realDeltaW = (c_loss - loss) / epsilon
			if (abs(realDeltaW - deltaW[i][j]) > 0.01):
				print("W gradient error!")
				return False
	for i in range(categories_size):
		tmpb = b.copy()
		tmpb[i][0] += epsilon
		c_loss, c_accuracy = test(X, Y, W, tmpb)
		realDeltab = (c_loss - loss) / epsilon
		if (abs(realDeltab - deltaB[i][0]) > 0.01):
			print("b gradient error!")
			return False
	print("check gradient end")


def update_batch(data, W, b, check=0):
	global categories_size, dimension, lam, rate
	deltaW = np.zeros((dimension, categories_size))
	deltaB = np.zeros((categories_size, 1))
	X = data[0]
	Y = data[1]
	N = len(X)
	for i in range(N):
		y = Y[i]
		x = X[i]
		x = x.reshape(dimension, 1)
		y = y.reshape(categories_size, 1)
		hat_y = softmax((np.dot(W.T, x) + b).reshape(categories_size)).reshape(categories_size, 1)
		I = np.zeros((categories_size, categories_size))
		for i in range(categories_size):
			I[i][i] = y[i]
		deltaW += np.dot(x, (y - hat_y).T)
		deltaB += y - hat_y
	deltaW = -1 / N * deltaW + 2 * lam * W
	deltaB = -1 / N * deltaB
	if check == 1:
		check_gradient(X, Y, W, b, deltaW, deltaB)
	W = W - rate * deltaW
	b = b - rate * deltaB
	return W, b


def softmax(x):
	tot = 0
	for i in range(len(x)):
		tot += pow(math.e, x[i])
	y = np.zeros(len(x))
	for i in range(len(x)):
		y[i] = pow(math.e, x[i]) / tot
	return y


def test(X, Y, W, b):
	N = len(X)
	global categories_size, dimension, lam
	loss = 0
	accuracy = 0
	for i in range(N):
		x = X[i]
		x = x.reshape(dimension, 1)
		y = Y[i]
		y = y.reshape(categories_size, 1)
		hat_y = softmax((np.dot(W.T, x) + b).reshape(categories_size)).reshape(categories_size, 1)
		loss += - 1 / N * np.dot(y.T, np.log(hat_y))[0][0]
		predict = np.argmax(hat_y.reshape(categories_size))
		if y[predict][0] == 1:
			accuracy += 1
	loss += lam * math.pow(np.linalg.norm(W), 2)
	accuracy /= N
	return loss, accuracy


def task2():
	dataset_train, dataset_test = handout.get_text_classification_datasets()
	N = len(dataset_train.data)
	global categories_size, dimension, lam, rate
	lam = 0.1
	rate = 0.1
	categories_size = len(dataset_train.target_names)
	count = {}
	dataset_train.data = deal_text(dataset_train.data)
	dataset_test.data = deal_text(dataset_test.data)
	for text in dataset_train.data:
		words = text.split(' ')
		for word in words:
			if word != "":
				if word not in count:
					count[word] = 0
				count[word] += 1
	vocabulary = {}
	dimension = 0
	for word in count:
		if count[word] >= 10:
			vocabulary[word] = dimension
			dimension += 1
	X, Y = deal_dateset(dataset_train, vocabulary)
	tX, tY = deal_dateset(dataset_test, vocabulary)
	batch_size = int(input("batch size ="))
	batches = [[X[k:k + batch_size], Y[k:k + batch_size]] for k in range(0, N, batch_size)]
	W = np.zeros((dimension, categories_size))
	b = np.zeros((categories_size, 1))
	lam = 0.001
	rate = 0.2
	epoch = 0
	his_loss = 100000
	min_loss = 100000
	max_test_accurcy = 0
	min_loss_epoch = 0
	train_loss_array = []
	test_loss_array = []
	rate_array = []
	while True:
		random.shuffle(batches)
		# update_batch(batches[0], W, b, 1)
		epoch += 1
		print("epoch: ", epoch)
		for batch in batches:
			W, b = update_batch(batch, W, b, 0)

		rate_array.append(rate)
		loss, accuracy = test(X, Y, W, b)
		print("train dataset loss=", loss, ", accuracy=", accuracy)
		# print("learn rate = ", rate)

		test_loss, test_accuracy = test(tX, tY, W, b)
		print("test dataset loss=", test_loss, ", accuracy=", test_accuracy)
		if test_loss < his_loss:
			rate = rate * 1.05
		else:
			rate = 0.1
		his_loss = test_loss

		train_loss_array.append(loss)
		test_loss_array.append(test_loss)
		if test_loss < min_loss:
			min_loss = test_loss
			min_loss_epoch = epoch
		elif epoch - min_loss_epoch > 20:
			break
		max_test_accurcy = max(max_test_accurcy, test_accuracy)

	print("max test dataset accurcy = ", max_test_accurcy)

	time = np.arange(epoch)
	plt.plot(time, train_loss_array)
	plt.plot(time, test_loss_array)
	plt.legend(["train loss", "test loss"])
	plt.show()
	plt.plot(time, rate_array)
	plt.legend(["learning rate"])
	plt.show()


if __name__ == "__main__":
	task = input("选择第一部分还是第二部分？(1/2)")
	if task == '1':
		task1()
	elif task == '2':
		task2()
