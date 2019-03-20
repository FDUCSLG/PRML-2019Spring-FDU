# -*- coding: utf-8 -*-
import os
#os.sys.path.append("/home/kengle/Documents/PRML/Assignment/PRML-Spring19-Fudan/assignment-1")
os.sys.path.append('../')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold

def histogram_method(sample_data, bins:int):
    plt.hist(sample_data, normed=True, bins = bins)
    plt.show()
    return

def histogram_method_ws(sample_data, bins:int):
    plt.hist(sample_data, normed=True, bins = bins)
    return

def kernel_function(point, sample_point, para_h):
    exp_value = math.exp(-0.5 * math.pow( (point-sample_point)/para_h ,2 ) )
    denominator = para_h * math.sqrt(2*math.pi)
    return exp_value/denominator

def kernel_density_estimation(points, sample_data, para_h):
    result = []
    for point in points:
        estimation = 0
        for sample_point in sample_data:
            estimation += kernel_function(point, sample_point, para_h)
        result.append(estimation/len(sample_data))
    return result

def kde_method(sample_data, para_h):
    left = min(sample_data)
    right = max(sample_data)
    # print(left,right)
    points_num:int = (int)((right-left)/0.1)
    x = np.linspace(left-1, right+1, points_num)
    plt.plot(x,kernel_density_estimation(x,sample_data,para_h))
    plt.show()

def knearest_neighbor_distance(point, sample_data,k:int=1):
    distances = []
    # mindis = abs(sample_data[0] - point)
    for sample_point in sample_data:
        dis = abs(sample_point - point)
        distances.append(dis)
        # mindis = min(mindis,dis)
    distances.sort()
    return max(distances[k-1]*2, 0.001/len(sample_data))
    #return max(distances[k-1]*2, 0.0000000001)

def knn_density_estimation(points, sample_data, k:int=1):
    result = []
    for point in points:
        mindis = knearest_neighbor_distance(point, sample_data,k)
        result.append( k/(mindis * len(sample_data)) )
    return result

def knn_method(sample_data, k:int=1):
    left = min(sample_data)
    right = max(sample_data)
    # print(left,right)
    points_num:int = (int)((right - left)/0.01)
    # points_num = 50
    x = np.linspace(left-1, right+1, points_num)
    plt.plot(x,knn_density_estimation(x,sample_data,k))
    plt.show()
    pass
    return

def show_all(sample_data, bins, para_h, k):
    left = min(sample_data)
    right = max(sample_data)
    # print(left,right)
    points_num:int = (int)((right-left)/0.1)
    points_num = 50
    x = np.linspace(left-1, right+1, points_num)
    
    plt.hist(sample_data, normed=True, bins = bins,label='Histogram')
    plt.plot(x,kernel_density_estimation(x,sample_data,para_h),'r', label='KDE')
    plt.plot(x,knn_density_estimation(x,sample_data,k), 'g', label='KNN')
    # plt.show()

def simple_test():
    sample_data = get_data(200)
    histogram_method(sample_data, 50)
    kde_method(sample_data, 0.2)
    knn_method(sample_data, 15)
    return 
# simple_test()

def test_showall():
    sample_data = get_data(200)
    show_all(sample_data, 50, 0.29, 20)
    plt.show()
    return
# test_showall()

def task1(bins:int=50, para_h:int=0.2, k:int=20):
    sample_data1 = get_data(100)
    sample_data2 = get_data(500)
    sample_data3 = get_data(1000)
    sample_data4 = get_data(10000)

    plt.subplot(3,2,1)
    plt.title("num_data=100")
    show_all(sample_data1, bins, para_h, k)
    
    plt.subplot(3,2,2)
    plt.title("num_data=500")
    show_all(sample_data2, bins, para_h, k)
    
    plt.subplot(3,2,5)
    plt.title("num_data=1000")
    show_all(sample_data3, bins, para_h, k)
    
    plt.subplot(3,2,6)
    plt.title("num_data=10000")
    show_all(sample_data4, bins, para_h, k)
    
    plt.show()
    return

def task2_try():
    sample_data = get_data(200)
    data_size = len(sample_data)
    for bins in range(2, 20 ,2):
        plt.title(bins)
        histogram_method(sample_data, bins)
    for bins in range(20,data_size, 20) :    
        plt.title(bins)
        histogram_method(sample_data, bins)
    return 

def task2():
    sample_data = get_data(200)
    data_size = len(sample_data)
    # Sqrt method
    plt.subplot(3,2,1)
    bins:int = int(math.sqrt(data_size))
    plt.title("Sqrt method:bins={}".format(bins))
    histogram_method_ws(sample_data, bins)
    
    # Sturges' formula: Better for Gaussian
    plt.subplot(3,2,2)
    bins:int=int(math.log(data_size,2)+1)
    plt.title("turges' formula:bins={}".format(bins))
    histogram_method_ws(sample_data,bins)
    
    # Rice rule: 
    plt.subplot(3,2,5)
    bins:int = int(2*math.pow(data_size,1/3))
    plt.title("Rice rule:bins={}".format(bins))
    histogram_method_ws(sample_data, bins)
    
    #Doane's formula
    plt.subplot(3,2,6)
    mean = np.mean(sample_data)
    sum1 = 0
    sum2 = 0
    for i in sample_data:
        sum1 += math.pow((i-mean),3)
        sum2 += math.pow((i-mean),2)
    sqrtb = sum1/math.pow(sum2,3/2)
    csqrtb = math.sqrt((6*data_size-12)/((data_size+1)*(data_size+3)))
    bins = int(1 + math.log(data_size,2) + math.log(1+(sqrtb/csqrtb),2))
    plt.title("Doane's formula:bins={}".format(bins))
    histogram_method_ws(sample_data, bins)
    plt.show()
    
    #Shimazaki and Shinomoto's choice
    minh = 0x7fffffff
    bins = 1
    sample_data.sort()
    left = sample_data[0]
    right = sample_data[data_size-1]
    for k in range(1,data_size+1):
        lst = [0 for i in range(k)]
        binsize = ((right-left)/k)+0.000001
        for data in sample_data:
            idx = math.floor(((data-left)/binsize))
            lst[idx] += 1
        mean = np.mean(lst)
        variance = np.var(lst)
        tmp = (2*mean - variance) / math.pow(binsize,2)
        if (tmp < minh):
            minh = tmp
            bins = k
    plt.title("Shimazaki and Shinomoto's choice:bins={}".format(bins))
    histogram_method(sample_data, bins)
    
    #Minimizing cross-validation estimated squared error
    mincv = 0x7fffffff
    bins = 1
    sample_data.sort()
    left = sample_data[0]
    right = sample_data[data_size-1]
    for k in range(1,data_size+1):
        lst = [0 for i in range(k)]
        binsize = ((right-left)/k)+0.000001
        for data in sample_data:
            idx = math.floor(((data-left)/binsize))
            lst[idx] += 1
        mean = np.mean(lst)
        variance = np.var(lst)

        n = data_size
        h = binsize
        tmp1 = 2/((n-1)*h)
        tmp2 = (n+1)/(n*n*(n-1)*h)
        Nk2 = []
        for num in lst:
            Nk2.append(num*num)
        tmp3 = sum(Nk2)
        tmp = tmp1 - tmp2*tmp3
        if (tmp < mincv):
            mincv = tmp
            bins = k
    plt.title("Minimizing cross-validation:bins={}".format(bins))
#    big_sample = get_data(10000)
#    plt.hist(big_sample, normed=True, bins = 50,label='Histogram')
#    plt.show()
    histogram_method(sample_data, bins)
    return

def cross_validation(train, test):
    len1 = len(train)
    len2 = len(test)
    n = min(len1,len2)
    sumdis = 0
    for i in range(n):
        sumdis += math.pow(train[i]-test[i],2)
    return sumdis/n
    
def task3_corss_validation(train,test,h,left,right):
    points_num:int = (int)((right-left)/0.1)
    x = np.linspace(left-1, right+1, points_num)
#    points_num:int = 500
#    x = np.linspace(21, 37, points_num)
    train_result = kernel_density_estimation(x,train,h)
    test_result = kernel_density_estimation(x,test,h)
    return cross_validation(train_result,test_result)
    

def task3():
    sample_data = get_data(500)
    n = len(sample_data)
    
#    for h in range(1,20):
#        plt.title("h={}".format(h))
#        kde_method(sample_data,h)
#    for h in range(1, 11):
#        plt.title("h={}".format(h/10))
#        kde_method(sample_data,h/10)
#    for h in range(30,60):
#        plt.title("h={}".format(h/100))
#        kde_method(sample_data,h/100)
    
    h = 1.06*np.std(sample_data) * math.pow(n,-1/5)
    plt.title("h={}".format(h))
    kde_method(sample_data,h)
    
    kf = KFold(n_splits = 5)
    mincv = 1000000.0
    minh = 1
    hlist = [(i/100) for i in range(1,201)]
    cvlist = []
    for h in range(1,201):
        h /= 100
        cv = 0.0
        for train,test in kf.split(sample_data):
            cv += task3_corss_validation([sample_data[i] for i in train],[sample_data[i] for i in test],h,min(sample_data),max(sample_data))
        cvlist.append(cv)
        if cv<mincv:
            mincv = cv
            minh = h
    h = minh
    plt.title("h={}".format(h))
    kde_method(sample_data,h)
    
    plt.plot(hlist,cvlist)
    
def task4_corss_validation(train,test,k,left,right):
    points_num:int = (int)((right-left)/0.01)
    x = np.linspace(left-1, right+1, points_num)
    train_result = knn_density_estimation(x,train,k)
    test_result = knn_density_estimation(x,test,k)
    return cross_validation(train_result,test_result)
    

def task4():
    sample_data = get_data(200)
    kf = KFold(n_splits = 2)
    mincv = 1000000.0
    mink = 1
    for k in range(1,31):
        #plt.title("K={}".format(k))
        #knn_method(sample_data, k)
        cv = 0.0
        for train,test in kf.split(sample_data):
            cv += task4_corss_validation([sample_data[i] for i in train],[sample_data[i] for i in test],k,min(sample_data),max(sample_data))
        if cv<mincv:
            mincv = cv
            # print(mincv)
            mink = k
    k = mink
    # k = int(math.sqrt(len(sample_data)))
    # k = 15
    plt.title("k={}".format(k))
    knn_method(sample_data,k)
# task2()
task3()

# task4()