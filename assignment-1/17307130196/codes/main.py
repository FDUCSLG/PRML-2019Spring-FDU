#FINAL GOAL: Use Nonparametric Methods to get probability distribution

#This is the first task of Assignment 1
#In this file, I will try to figure out what will happen with different numbers of samples.

#Histogram Method, Kernel Method and the Nearest Neighbor Method will be showed here.
#-------------
# And each function are called Hist(), Kerner(), and KNN()
import Histogram_Method
import Nearest_Neighbor_Method
import Kernel_Method
def main(c,num,bins,K,H,N):
    if(c=='H'): Histogram_Method.Hist(N,bins)
    elif(c=='K'): Kernel_Method.Kernel(num,N,H)
    else: Nearest_Neighbor_Method.KNN(num,N,K)

# bins is the number of bins in Histogram Method
# H is the parameter in Kernel Method of the volume
# K is the key parameter in Nearest Neighbor Method
# num is the number of test in Kernel and Nearest Neighbor Method