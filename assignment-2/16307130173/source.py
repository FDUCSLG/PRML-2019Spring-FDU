import sys
sys.path.append('..')
from handout import get_text_classification_datasets, get_linear_seperatable_2d_2c_dataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import string

text_train, text_test = get_text_classification_datasets()

batch_size = 32
num_of_class = 4

class LeastSquare(object):
    def getmat(self, ds):
        tt = []
        for xi in ds.X:
            tt.append([1] + [xx for xx in xi])
        self.x_mat = np.mat(np.array(tt))
        self.t_mat = np.mat(np.array([2 * int(yi > 0) - 1 for yi in ds.y]))

    def train(self, ds):
        self.getmat(ds)
        self.w = np.array((((self.x_mat.T * self.x_mat).I * self.x_mat.T) * self.t_mat.T).T)
        self.w = self.w[0]
                
        return self
    
    def dots(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.dots(X) >= 0, 1, -1)

    def plot(self, ds):
        class_1 = {"x": [], "y": []}
        class_2 = {"x": [], "y": []}
        y = self.predict(ds.X)

        for xi, yi in zip(ds.X, y):
            if yi == 1:
                class_1["x"].append(xi[0])
                class_1["y"].append(xi[1])
            else:
                class_2["x"].append(xi[0])
                class_2["y"].append(xi[1])

        linex = np.arange(-2, 2)
        liney = -self.w[1] / self.w[2] * linex - self.w[0] / self.w[2]
        
        plt.scatter(class_1["x"], class_1["y"])
        plt.scatter(class_2["x"], class_2["y"])
        plt.plot(linex, liney)
        plt.show()

        return ds.acc([int(yi > 0) for yi in y])
        
class Perceptron(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, ds):
        self.w = np.zeros(ds.X.shape[1] + 1)

        errors = 1
        while errors != 0:
            errors = 0
            for xi, yi in zip(ds.X, ds.y):
                if yi == 0:
                    yi = -1
                update = self.learning_rate * 0.5 * (yi - self.predict(xi))
                errors += int(update != 0)
                
                self.w[0] += update
                self.w[1:] += update * xi
            
        return self

    def dots(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.dots(X) >= 0, 1, -1)

    def plot(self, ds):
        class_1 = {"x": [], "y": []}
        class_2 = {"x": [], "y": []}
        y = self.predict(ds.X)

        for xi, yi in zip(ds.X, y):
            if yi == 1:
                class_1["x"].append(xi[0])
                class_1["y"].append(xi[1])
            else:
                class_2["x"].append(xi[0])
                class_2["y"].append(xi[1])

        linex = np.arange(-2, 2)
        liney = -self.w[1] / self.w[2] * linex - self.w[0] / self.w[2]
        
        plt.scatter(class_1["x"], class_1["y"])
        plt.scatter(class_2["x"], class_2["y"])
        plt.plot(linex, liney)
        plt.show()

        return ds.acc([int(yi > 0) for yi in y])

class Word2Vec(object):
	def __init__(self, ds, mincount):
		self.dictionary = {}
		self.D = self.getDict(ds, mincount)
		
	def trans(self, st):
	
		tt = st.lower()
		for j in string.punctuation:
			tt = tt.replace(j, '')
		for j in string.whitespace:
			tt = tt.replace(j, ' ')							
			
		return tt.split(' ')
		
	def getDict(self, ds, mincount):						
		ds_len = len(ds)
		tmp_map = {}
		for i in range(ds_len):				
			dsi = self.trans(ds[i])
			
			for li in dsi:
				if li != "":					
					if li not in tmp_map.keys():					
						tmp_map[li] = 0
					tmp_map[li] += 1
		D = 0
		for key, val in tmp_map.items():
			if val >= mincount:				
				self.dictionary[key] = D
				D += 1
		return D
	
	def getVec(self, string_list, tar):
		N = len(string_list)	
		text_vector = np.zeros((N, self.D))

		i = 0
		for si in string_list:
			st = si.lower()
			for sj in string.punctuation:
				st = st.replace(sj, '')
			for sj in string.whitespace:
				st = st.replace(sj, ' ')
			item = st.split(' ')
					
			for sj in item:				
				if sj in self.dictionary.keys():
					text_vector[i][self.dictionary[sj]] = 1
			i += 1
			
		return Dataset(text_vector, tar)

class Softmax(object):
	def __init__(self, class_num):
		self.W = None
		self.losses = []
		self.class_num = class_num

	def loss(self, W, ds, reg = 0.001):
		N, D = ds.X.shape
		dW = np.zeros_like(W)	
		
		X_norm = np.hstack((np.ones((N, 1)), ds.X))
		scores = X_norm.dot(W)
		
		correct_score = scores[range(N), ds.y].reshape(-1, 1)
		exp_sum = np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
				
		loss = np.sum(np.log(exp_sum) - correct_score)
		loss = loss / N + reg * np.sum(W * W)

		temp = np.exp(scores) / exp_sum
		temp[range(N), ds.y] -= 1
		dW = X_norm.T.dot(temp) / N + reg * W
		return loss, dW

	def train(self, ds, learning_rate = 0.1, batch_size = 32, num_iter = 1000, reg = 0.001):
		N, D = ds.X.shape
		X_norm = np.hstack((np.ones((N, 1)), ds.X))
		
		if self.W is None:
			self.W = 0.001 * np.random.randn(D + 1, self.class_num)
		
		for i in range(num_iter):
			rand_index = np.random.choice(N, batch_size, replace = True)			
			loss, dW = self.loss(self.W, Dataset(ds.X[rand_index], ds.y[rand_index]), reg)

			self.losses.append(loss)
			self.W -= learning_rate * dW

	def predict(self, X):
		if self.W is None: 
			return "Not Trained Yet"
		X_norm = np.hstack((np.ones((X.shape[0], 1)), X))
		pred = np.argmax(X_norm.dot(self.W), axis = 1)
		return pred

	def accuracy(self, ds):
		y_pred = self.predict(ds.X)
		return np.mean(y_pred == ds.y)
	
	def plot(self, plt):
		loss_len = len(self.losses)
		plt.plot(range(loss_len), self.losses)
		return plt

#model1 = LeastSquare().train(text_train)
#print(model1.plot(text_train))

#model2 = Perceptron(1).train(text_train)
#print(model2.plot(text_train))

premodel = Word2Vec(text_train.data, 10)

ds_train = premodel.getVec(text_train.data, text_train.target)
ds_test = premodel.getVec(text_test.data, text_test.target)

for i in [1, batch_size, ds_train.X.shape[0]]:
	softmax = Softmax(num_of_class)
	softmax.train(ds_train, batch_size = i, num_iter = 1000)
	print(softmax.accuracy(ds_test))
	softmax.plot(plt).show()