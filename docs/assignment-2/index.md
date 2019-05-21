## <center>Pattern Recognition and Machine Learning</center>

### <center>Fudan University / 2019 Spring</center>

<center>Assignment 2</center>

In this assignment you are going to explore several well-know linear classification methods such as perceptron and logistic regression, and a realistic dataset will be provided for you to evaluate these methods.

#### Description

##### Part 1

To start with, we consider the least square model as an extension for the linear regression model to the problem of classification, and also the perceptron algorithm. You are given a simple linearly separable dataset containing 2 classes of points on a 2D plane, and you should build a model to correctly classify them.

You are provided with a function `gen_linear_seperatable_2d_2c_dataset` in the handout file, you should import it to your solution as in the first assignment.

![dataset](./lin.png)

Requirements

Namely speaking, you should use least square model and the perceptron algorithm to learn two models to separate the dataset, and report the accuracy after you have learned the model.

Since the two models are linear, you should be able to draw a decision line on top of the dataset to visually show how can you separate the dataset. Include this plot in you report.

Part 2

In this part of the assignment, you are required to use logistic regression to do a simple text classification task. A function `get_text_classification_datasets` is also provided, which returns the training and testing dataset, and you could explore the dataset by looking more closely to what it returns. Note that calling the function for the first time will takes several minutes to cache the dataset to your disk (the path is `../../..` which is parallel to the root of the project directory).

Text classification is about to classify a given document with its corresponding genre, e.g. sports, news, health etc. To represent the document in a convenient way to be processed by the logistic regression, we could represent the document in a multi-hot style vector.

Firstly you will have to tokenize the document into words (or a list of strings), you should firstly ignore all the characters in `string.punctuation` and then make all `string.whitespace` characters a space , and then split the document by all the spaces to have a list of strings.  Subsequently, you should convert all characters into lowercase to facilitate future process. Then you build a vocabulary on all the words in the training dataset, which simply maps each word in the training dataset to a number. For example, 

```python
docs_toy = [
"""
Hi!

How are you?

""",
"""
Do you have a dog?
"""
]
```

When tokenized you will have

```python
['hi', 'how', 'are', 'you'],
['do', 'you', 'have', 'a', 'dog']
```

and your vocabulary will look like

```python
{'a': 0, 'are': 1, 'do': 2, 'dog': 3, 'have': 4, 'hi': 5, 'how': 6, 'you': 7}
```

and you use the vocabulary to map the tokenized document to a multi-hot vector!

```python
[0. 1. 0. 0. 0. 1. 1. 1.]
[1. 0. 1. 1. 1. 0. 0. 1.]
```

as you could verify this is the representation from the above two document.

In practice, the vocabulary dictionary is quite large, which may cause the size of multi-hot vector exceeds the memory limits! To address this problem, you can set a frequency threshold  `min_count` and only consider the words which occur at least `min_count` times in the overall training set. For this problem, `min_count = 10` is suitable.

Once you could represent the document in vectors (and also the category of the document in one-hot representation), then you can use the logistic regression!

Logistic regression is a kind of generalized linear model, the major (or only?) difference between logistic regression and least square is that in logistic regression we use a non-linear function after the linear transformation to enable probabilistic interpretation for the output. For binary classification, the logistic  sigmoid function
$$
\sigma(a)={1\over 1 + \exp{(-a)}}
$$
transforms the unbounded prediction $a\in(-\infty,+\infty)$ from the output of the linear model to a bounded interval $(0, 1)$, which could be interpreted as the probability of the prediction.

Note that logistic sigmoid function is just a special case of the following softmax function. Given a vector $\mathbf{z}$ of the dimension $k​$, the softmax function computes the vector with the following components:
$$
 \text{softmax}\left(\mathbf{z}\right)_i = \frac{e^{z_i}}{\sum_{j=1}^ke^{z_j}} 
$$


In other words, the softmax function first exponentiates the vector (elementwise) and then normalizes it such that all the components would add up to 1. The resulting vector can be interpreted as a probability distribution over the number of classes. We can then make prediction through selecting a class with the maximum associated probability:

$$
 \hat{y}_{pred} = \underset{j \in 1..k}{\text{argmax}}\left[\text{softmax}\left(\mathbf{z}\right)\right] 
$$


One interesting property of the softmax function is that it is invariant to constant offsets, i.e. $\text{softmax}\left(\mathbf{z}\right) = \text{softmax}\left(\mathbf{z} + \mathbf{c}\right)$, where $\mathbf{c}$ is a broadcasted vector of equal constant values. This means you could (and perhaps should) subtract the maximum from $\textbf{z}$ before $\text{softmax}$ to stabilize the numerical calculation.

To sum up, for logistic regression the predicted probability distributions are computed as:

$$
\mathbf{\hat{y}} = \text{softmax}\left(W^{T}\mathbf{x} + \mathbf{b}\right)
$$
Because the existence of the softmax function, we could not write a closed solution for the optimization, therefore we will use gradient descent to optimize a loss function, as is typically done in machine learning, we minimize a loss function on top of the prediction of the linear model. Concretely we use the cross entropy loss function defined as
$$
\mathcal{L} = - \frac{1}{N}\sum_{n=1}^N y_n\log \hat{y}_n \rightarrow \min_{W, b}
$$
where $\hat{y}_n$ is the predicted probability of the correct class $y_n$ for the $n$-th training example and $N$ is the total number of training examples. In order to reduce overfitting, an additional term penalizing large weights is added to the loss function as:

$$
\mathcal{L} = - \frac{1}{N}\sum_{n=1}^N y_n\log \hat{y}_n + \lambda \Vert W\Vert ^2 \rightarrow \min_{W, b}
$$
Stochastic gradient descent starts by taking the gradient of each parameter with respect to the loss, and then update the parameter with the gradient. Namely
$$
W_{ij} \leftarrow W_{ij} - \alpha\frac{\partial \mathcal{L}}{\partial W_{ij}} \\
b_{i} \leftarrow b_{i} - \alpha\frac{\partial \mathcal{L}}{\partial b_{i}}
$$

where $\alpha$ is a hyperparameter called learning rate that adjusts the magnitude of the weight updates.

Since the loss function of logistic regression is a convex function (as you could verify by differentiating the loss function twice), you are promised to get to the global minimum of the loss function with gradient descent.

To conclude, the requirements are:

1. Implement the preprocess pipeline to transform the documents into multi-hot vector representation, and also the targets into one-hot representation, indicate how you implemented them in your report
2. Differentiate the loss function for logistic regression, write down how you compute ${\partial \mathcal{L}\over\partial W_{i,j}},{\partial \mathcal{L}\over \partial b_i}$, and then implement the calculation in vectorized style in numpy (meaning you should not use any explicit loop to obtain the gradient). Answer the two questions in your report: (1) Sometimes, to overcome overfitting, people also use L2 regularization for logistic regularization, if you add the L2 regularization, should you regularize the bias term? (2) how do you check your gradient calculation is correct?
3. Finish the training for the logistic regression model on the training dataset, include the plot for the loss curve you obtained during the training of the model. Answer the questions: (1) how do you determine the learning rate? (2) how do you determine when to terminate the training procedure?
4. Somethings, other than doing a full batch gradient descent (where you take into consideration all the training data during the calculation of the gradient), people use stochastic gradient descent or batched gradient descent (meaning only one sample or several samples per update), you should also experiment with the other 2 ways of doing gradient descent with logistic regression. Answer the questions: (1) what do you observe by doing the other 2 different ways of gradient descent? (2) can you tell what are the pros and cons of each of the three different gradient update strategies?
5. Report your result for the three differently trained model on the test dataset. <p style="color:red">Do not peek into the test dataset before this requirement!</p>