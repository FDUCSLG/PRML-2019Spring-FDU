import os
os.sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from handout import get_data
import math
def task1(N):
	
	sameple_data = get_data(N)
	#print(sameple_data)
	bins = int(math.sqrt(N)) + 1
	
	plt.hist(sameple_data, bins=bins, normed=True)
	plt.title("data size=" + str(N))
	plt.show()
	
def Guassian_kernel_function(data,h,x):
	sum = 0
	for x_i in data:
		sum += pow(math.e,-((x - x_i)**2)/2/(h**2)) / math.sqrt((2 * math.pi * (h**2)))
	return sum / len(data)
def KNN(data, K, x):
	l = []
	for x_i in data:	
		l.append(abs(x - x_i))
	l.sort()
	#print(K, len(data), l[K])
	return K / len(data) / (l[K] * 2)


def cross_validation(data, h):
	K = 5
	N = len(data)
	m = N // K
	batch = [data[i:i+m] for i in range(0,N,m)]
	sum = 0
	for i in range(K):
		train_data = []
		for j in range(K):
			if i != j:
				train_data.extend(batch[j])
		for x in batch[i]:
			#print(Guassian_kernel_function(train_data,h,x))
			sum += math.log(Guassian_kernel_function(train_data,h,x))
	#print(sum)
	return sum / K
	
# use Silverman’s rule  to determine the h
# h = (4/(3n))^(0.2) * sqrt(variance)
def silverman(data):
	tot = 0
	for x in data:
		tot += x
	average = tot / N
	variance = 0
	for x in data:
		variance += ((x - average)**2) / (N-1)
	
	h = pow(4/(3*N),0.2) * math.sqrt(variance)
	return h
	
def task2(N):
	sample_data = get_data(N)
	max_data = max(sample_data)
	min_data = min(sample_data)
	rang = max_data - min_data
	max_limit = max_data + rang / 2
	min_limit = min_data - rang / 2
	
	
	#h = silverman(sample_data)
	#h = 2
	h = 0.3
	if N<=100:
		max_value = -100000
		x = np.arange(0.1,2,0.01)
		y = []
		#print(x)
		for x_i in x:
			tmp = cross_validation(sample_data,x_i)
			y.append(tmp)
			#print(x_i,tmp)
			if (max_value < tmp):
				max_value = tmp
				h = x_i
		plt.plot(x, y)
		plt.title("likelihood function")
		plt.show()
	print("h=",h)
	
	
	
	
	x = np.linspace(min_limit, max_limit, num=1000)
	y = []
	for i in x:
		y.append(Guassian_kernel_function(sample_data,h,i))
	plt.plot(x, y)
	plt.title("data size=" + str(N))
	plt.show()
def task3(N):
	c = input("是否选择多种K值画图？(y/n)")
	sample_data = get_data(N)
	max_data = max(sample_data)
	min_data = min(sample_data)
	rang = max_data - min_data
	max_limit = max_data + rang / 2
	min_limit = min_data - rang / 2
	x = np.linspace(min_limit, max_limit, num=1000)
	if (c == 'n'):
		K = 10
		y = []
		for x_i in x:
			y.append(KNN(sample_data, K, x_i));
		
		plt.plot(x, y)
		plt.title("data size=" + str(N))
		plt.show()
	if (c == 'y'):
		K = [5,10,30]	
		for k_i in K:
			y = []
			for x_i in x:
				y.append(KNN(sample_data, k_i, x_i));
			plt.plot(x, y)
		#plt.legend(map(lambda x:"k="+str(x),K))
		plt.title("data size=" + str(N))
		plt.legend(["K=5","K=10","K=30"])
		plt.show()
if __name__ == "__main__":
	method = input("输入1为直方图，2为核密度估计，3为k近邻方法：")
	N = int(input("输入数据大小："))
	if (method == '1'):
		task1(N)
	elif (method == '2'):
		task2(N)
	elif (method == '3'):
		task3(N)