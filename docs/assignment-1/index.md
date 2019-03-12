## <center>Pattern Recognition and Machine Learning</center>

### <center>Fudan University / 2019 Spring</center>

<center>Assignment 1</center>

Note: This is the first assignment for this course, you should read the [guidelines](https://github.com/ichn-hu/PRML-Spring19-Fudan/blob/master/README.md) and consent to the agreement before doing it.

This assignment will due on Wed 20 before the course start, for instructions of submission, please also see the guidelines.

#### Description

In this assignment, you are going to use the three non-parametric density estimation algorithms (namely histogram method, kernel density estimate and the nearest neighbor method) to estimate the distribution of the given data set. Since this is the first assignment of this course, it is designed to be simple for you to warm up.

More concretely, you are given a data set consists of 10000 samples from an unknown 1-D distribution, you can access the data set by a simple function provided for you in `assignment-1/handout`

```python
def get_data(num_data:int = 100) -> List[float]:
    """
    Please use this function to access the given distribution, you should provide an int
    `num_data` to indicate how many samples you want, note that num_data must be no
    larger than 10000
    """
    assert num_data <= 10000
    return list(sampled_data[:num_data])
```

you should estimate and plot the distribution using `matplotlib` with the three desired algorithms, note that in `assignment-1/16307130177/source.py` a solution with histogram is provided, you could use it as a start point.


#### Report Requirements

To report your estimation for this distribution, you should play with the three given algorithms with respect to the following requirements.

* (10%) For all three algorithms, you should vary the number of data used, let's say you could use 100, 500, 1000, 10000 to see what will happen for your estimation, you don't have to report 3 * 4 = 12 plots just make an empirical assertion about how does the number of data influence the quality of the estimation. In the rest part of the requirements, if not specified, use `num_data=200` for exploration (or other number you think is better for your exploration, please state clear and keep consistent if so).

* (20%) For histogram estimation, you could vary the number of bins used to locate the data samples to see how this parameter affect the estimation. Please answer: how could you pick the best (or good) choice for this number of bins?

* (30%) For kernel density estimation, you should try the Gaussian kernel
  $$
  p(\mathbf{x})={1\over N}\sum_{n=1}^N{1\over (2\pi h^2)^{1/2}}\exp\left\{-{||\mathbf{x}-\mathbf{x}_n||\over 2h^2}\right\},
  $$
  Please also try to tune $h$ to see what will happen, answer if you have a clue of how to choose $h$, plot the best estimate you could achieve with `num_data=100`.

* (40%) For nearest neighbor methods, you should vary $K$ in $p(\mathbf{x})={K\over NV}$ to see the difference, you are encouraged to plot an illustration as Figure 2.26 in the text book, to plot the true distribution, see `GaussianMixture1D.plot` in the handout for more details. Additionally, please show that the nearest neighbor method does not always yield a valid distribution, i.e. it won't converge to 1 if you sum the probability mass over all the space (in our case $(-\infty, \infty)$), you could show this either empirically or theoretically.

