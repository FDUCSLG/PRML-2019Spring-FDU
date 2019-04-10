#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
def Gaussian(x,x0,h):
	return math.exp(-((x-x0)*(x-x0)/(2*h*h)))/math.sqrt(2*(math.pi)*h*h)    


# In[2]:


import os
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt


# In[109]:


def histogramMethod(num_sample,num_bins):
    sample_data = get_data(num_sample)
    plt.hist(sample_data,normed = True, bins = num_bins)
    plt.show()
histogramMethod(10000,200)


# In[108]:


def KDE(num_sample,h):
    sample_data = get_data(num_sample)
    xlist = np.linspace(min(sample_data),max(sample_data),2*num_sample)
    ylist = np.zeros_like(xlist)
    i = 0
    for x in xlist:
        sum = 0
        for x_compare in sample_data:
            sum += Gaussian(x,x_compare,h)
        ylist[i] = sum/num_sample
        i += 1
    plt.plot(xlist,ylist)
    plt.xlabel("x")
    plt.ylabel("y")
KDE(100,0.5)
plt.show()


# In[107]:


from sklearn.model_selection import KFold
def KFold_cross_validation_KDE(num_sample):
    sample_data = get_data(num_sample)
    print(sample_data)
    kf = KFold(n_splits = 3)
    h_test = 0
    minCV = 10000000
    h_ideal = 0
    for i in range(0,100):
        h_test += 0.01
        CV = 0
        print(i)
        for train_index,test_index in kf.split(sample_data):
            print("_______________")
            train = []
            test = []
            for idx in train_index:
                train.append(sample_data[idx])
            for idx in test_index:
                test.append(sample_data[idx])
#             print(train)
#             print(test)
            xlist = np.linspace(min(sample_data),max(sample_data),200)
            y_train = np.zeros_like(xlist)
            y_test = np.zeros_like(xlist)
            j = 0
            for x in xlist:
                sum = 0
                for x_compare in train:
                    sum += Gaussian(x,x_compare,h_test)
                y_train[j] = sum/len(train)
                j += 1
            j = 0
            for x in xlist:
                sum = 0
                for x_compare in test:
                    sum += Gaussian(x,x_compare,h_test)
                y_test[j] = sum/len(test)
                j += 1
            MSE = 0
            for j in range(0,len(xlist)):
                MSE += math.pow(y_train[j] - y_test[j],2)
            CV += MSE/len(xlist)
            print(CV)
        
        if(CV < minCV):
            minCV = CV
            h_ideal = h_test
    print(h_ideal)
    KDE(num_sample,h_ideal)
KFold_cross_validation_KDE(100)


# In[121]:


def KDEtestSample10000():
    sample_10000 = get_data(10000)
    xlist = np.linspace(min(sample_10000),max(sample_10000),200)
    ylist = np.zeros_like(xlist)
    index = 0
    for s in sample_10000:
        index = int(((s-min(sample_10000))/(max(sample_10000)-min(sample_10000)))*200)
        ylist[min(index,199)] += 1
    for i in range(0,200):
        ylist[i] /= 10000
        ylist[i] /= (max(sample_10000)-min(sample_10000))/200
    sum1 = 0
    for i in range(0,200):
        sum1 += ylist[i]
    print(sum1)
    plt.plot(xlist,ylist)
    plt.xlabel("x")
    plt.ylabel("y")
    h_ideal = 0
    h_temp = 0
    min_error = 1000000
    for i in range(0,100):
        h_temp += 0.01
        sample_data = get_data(100)
        xlistKDE = np.linspace(min(sample_data),max(sample_data),200)
        ylistKDE = np.zeros_like(xlistKDE)
        j = 0
        for x in xlistKDE:
            sum = 0
            for x_compare in sample_data:
                sum += Gaussian(x,x_compare,h_temp)
            ylistKDE[j] = sum/100
            j += 1
        error = 0
        for j in range(0,200):
            error += (ylist[j]-ylistKDE[j])*(ylist[j]-ylistKDE[j])
        if error < min_error:
            min_error = error
            h_ideal = h_temp
    KDE(100,h_ideal)
    print(h_ideal)
KDEtestSample10000()
plt.show()


# In[104]:


def knn(num_sample,K):
    sample_data = get_data(num_sample)
    xlist = np.linspace(min(sample_data),max(sample_data),2*num_sample)
    ylist = np.zeros_like(xlist)
    i = 0
    integration = 0
    for x in xlist:
        j = 0
        dis_list = np.zeros_like(sample_data)
        for x_compare in sample_data:
            dis_list[j] = abs(x_compare - x)
            j += 1
        dis_list.sort()
        ylist[i] = K/(num_sample*2*max(dis_list[K-1],0.001))
        integration += (max(sample_data) - min(sample_data))*ylist[i]/(2*num_sample)
        i += 1
    print(integration)
    plt.plot(xlist,ylist)
    
    plt.xlabel("x")
    plt.ylabel("y")
knn(100,6)
plt.show()


# In[91]:


from handout import GaussianMixture1D
import scipy.stats as ss
np.random.seed(0)
gm1d = GaussianMixture1D(mode_range=(0, 50))
def trueDistributionCopied():
    data = gm1d.sample([2000])
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)
    
ax = plt.subplot(321)
knn(200,1)
trueDistribution()

ax = plt.subplot(322)
knn(200,6)
trueDistribution()

ax = plt.subplot(323)
knn(200,15)
trueDistribution()

ax = plt.subplot(324)
knn(200,30)
trueDistribution()

ax = plt.subplot(325)
knn(200,50)
trueDistribution()

ax = plt.subplot(326)
knn(200,70)
trueDistribution()

plt.show()


# In[94]:




# In[ ]:




