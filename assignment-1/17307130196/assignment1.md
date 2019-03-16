# Assignment 1 of PRML

## Introduction

In this assignment, four requirements are listed below:

- What will happen when we varies the number of data used?
- How the number of bins of Histogram affects the estimation?
- Try different *h* for Kernel Method
- Vary *K* in Nearest Neighbor Method

  And they will be solved case by case. But before that, I will give a brief *summery* of the details I will discuss in this paper. And after that, fake codes for the Kernel Method and NNM will be presented. (As the function of Histogram Method is provided by `matplotlib`, I will not show it here.)

**Method Used**: Histogram Method, Kernel Method and the Nearest Neighbor Method

**Goal**: Estimate the distribution of a given dataset. And try to figure out how the parameters of these methods affects the result.

- Requirement One
  - How the number of samples affects the result?
  - Using suggested numbers for testing the three methods.
  - Find some more details in #TODO
- Requirement Two
  - Consider Histogram Method
  - Figure out how the number of bins affects the result
  - How to pick out the best choice for the number?
- Requirement Three
  - Figure out how the volume of the box affects the result
  - And how to choose a better *h* for estimation
  - Using a dataset of 100 samples. Try to get a best figure by choosing a suitable *h*
- Requirement Four
  - Plot an illustration
  - Show the method isn't always valid
    - Empirical way
    - Theoretical way

**Note**: Although the formulars used to compute the probability will be showed, the deduction will be left out.

## Fake Code

### Kernel Method

```
def KernelGaussian(target,dataset,paras): 
# target is the point we're computing the probability
# dataset is the sample data provided
# paras include the all the parameters of the methods
	SET sum 0
	FOR EACH data IN dataset:
		ADD result TO sum 
		#result is the computing result of the data according to the Gaussian Func
	ALTER sum ACCORDING TO paras

def Kernel(N,num):
	GENERATE sample_data ACCORDING TO N
	GENERATE test_points ACCORDING TO num
	FOR EACH test_point:
		GET the output of the point BY KernelGaussian
	PLOT
```

### Nearest Neighbor Method

```
def KNN_Pro(target,dataset,N):
#dataset here is ordered by ascending
	SET flag_data TO the first data no less than target
	SET ct 0 #ct marks the number of points met
	SET flag_l TO data before flag_data
	SET flag_r TO flag_data
	WHILE ct < K:
		FIND the closer point
		ALTER flag ACCORDINGLY
	SET V TO flag_r-flag_l
	COMPUTE BY FORMULAR WITH K, V, N

def KNN(N,num,K):
GENERATE sample_data ACCORDING TO N
	GENERATE test_points ACCORDING TO num
	FOR EACH test_point:
		GET the output of the point BY KNN_Pro
	PLOT
```



## Assignment One

 