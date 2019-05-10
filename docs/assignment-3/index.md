## <center>Pattern Recognition and Machine Learning</center>

### <center>Fudan University / 2019 Spring</center>

<center>Assignment 3</center>

In this assignment you are going to implement a RNN (namely LSTM) for generating Tang poetry. This assignment description will outline the landscape for you to know how to do it!

#### Description

In the previous assignment, you've already implemented back-propagation of gradients with numpy, you must have had a lot of fun playing with it. Although nowadays autograd tools like Tensorflow and PyTorch are pervasive, and people rarely write deep neural networks without them, not only because they provided great convenience over gradient computation, but also could they leverage GPUs for amazingly fast training, knowing the details under the hook should be very beneficial if you want to dive deeper into deep learning and these are very frequently asked during interviews.

##### Part 1, Differentiate LSTM, 20%

In the course, we talked about Recursive Neural Network, and one of its mostly used variation, LSTM (Long-Short Term Memory [^Hochreiter & Schmidhuber (1997)]) network. To remind your how LSTM works, the LSTM unit $ \mathbf{h}_t=LSTM(\mathbf{h}_{t-1}, \mathbf{x}_t)$ processing the input in the following manner under the hook

$$
\begin{align}
\mathbf{z} &= [\mathbf{h}_{t-1},\mathbf{x}_t]\\
\mathbf{f}_t & = \sigma(W_f \cdot \mathbf{z} + b_f) \\
\mathbf{i}_t & = \sigma(W_i \cdot \mathbf{z} + b_i) \\
\bar{C}_t & = tanh(W_C \cdot \mathbf{z} + b_C) \\
C_t & = \mathbf{f}_t * C_{t-1} + \mathbf{i}_t * \bar{C}_t \\
\mathbf{o}_t & = \sigma(W_o \cdot \mathbf{z} + b_t) \\
\mathbf{h}_t &= \mathbf{o}_t * tanh(C_t) \\
\end{align}
$$
where $\cdot$ stands for matrix multiplication, $*$ for dot production and $[\cdot, \cdot]$ for vector concatenation. Note $W_{*}$ and $b_{*}$ are parameters of the LSTM that is the same throughout all steps.

Also note that here the input is a vector while in your implementation, please use batched input as matrix multiplication on matrix is the same as multiplying vectors concatenated horizontally.

For language modeling, we use LSTM to predict the next word or character at each step. For example, if we have a sentence $s_1, s_2,\cdots, s_n$ for the input at each step for the LSTM, the output at each step should be $s_2,s_3,\cdots, s_n, \text{EOS}$ where EOS stands for end of sentence. To obtain an prediction from LSTM, we first create an vocabulary $V$ to map each word to an integer which is an ordered set that contains all the word in your training dataset, and then we could map each integer $i$ to an vector $\mathbf{x}_i$ which will be the input for the LSTM. Then we rely on the hidden vector, at each step t, we can use a linear transformation $\mathbf{a}_t = W\mathbf{h}_t + b$  where $\mathbf{a}_t$ is a vector of size $|V|$. Because linear transformation results in value that is unbounded, to make prediction a probability distribution, we first take exponential and then normalize it with the sum e.g. take the softmax of $\mathbf{a}_t$, $\mathbf{y}_t^{(i)}=\frac{\exp(\mathbf{a}_t^{(i)}/\tau)}{\sum_{j=0}^{|V|-1}\exp(\mathbf{a}_t^{(j)}/\tau)}$, where $\tau$ is the **temperature** term that is usually 1, you will encounter this term later. As we've learned from the previous assignment, we could use cross entropy loss to urge the prediction to be the next word $s_{t+1}$, and we could try to minimize the average loss to provide the training signal for the network.


Requirements

1. Differentiate one step of LSTM with respect to $\mathbf{h}_t$ for $\mathbf{f}_t, \mathbf{i}_t, \mathbf{i}_t, \bar{C}_t, C_t, C_{t-1},\mathbf{o}_t, \mathbf{h}_{t-1}, \mathbf{x}_t$. i.e. $\frac{\partial \mathbf{h}_t}{\partial \mathbf{f}_t}$, include your formalization and derivation in your report. 10%

2. Describe how can you differentiate through time for the training of an LSTM language model for sentence $s_1,s_2,\cdots, s_n$. 10%

##### Part 2, Autograd Training of LSTM, 80%

In this part you are going to implement an LSTM to build a language model to generate Tang poetry.

You are given a small dataset containing some Tang poems, you first split the dataset to a training dataset and development dataset, we would recommend an 80% and 20% split. Then you create a vocabulary containing all the words (or characters, but we stick to use words to refer to them) in the training dataset, be aware that you might want to insert a new word `EOS` and a special token  `OOV` for unknown word (or known as out-of-vocabulary word). To process the dataset, you should transform the poems into a sequence of integer representing words in the vocabulary. Then you could randomly crop the sequence into batches of short sequences for the training of the LSTM. Note that at each step a single input of the LSTM should be a vector, we should create a mapping from integers to vectors, this step is also known as **embedding** in NLP. And follow the previous discussion we could come to a loss function that could provide gradient to the parameters and also the embedding (as you could either fix the embedding to its initialization or update it with the gradient).

As the model is pretty clear here, you should include the hyperparameter and training setting your are using in your report. They are

- Vocabulary size, $|V|$
- Batch size, $bs$
- Sentence length, $sl$
- Hidden size, i.e. the length of the hidden vector of the LSTM, $hs$
- Input size, i.e. the length of the input vector for the LSTM, $is$

The training of the model stops when it could not get better in predicting the next word on the development dataset, which could be evaluated by *perplexity* 
$$
\begin{align}
PP(S)&=P(s_1s_2\cdots s_N|LM)^{-1/N}\\
&=\sqrt[N]{{1\over \prod_{i=t}^{N}\mathbf{y}_t^{(s_{t+1})}}}
\end{align}
$$
The perplexity should be evaluated on the whole development dataset, which is to split the dataset by length $sl$ which is the sentence length used in the training stage, and then evaluate the average perplexity on all the split sentences. Use early stop when perplexity don't improve.

To generate a Tang poem once you got the model trained, you could first sample a word to start and then use it as input to the LSTM, and them sample from the output of the LSTM and in turn send the generated word into the LSTM to have the next word generated. To allow more variation, sometimes people use a **temperature term $\tau$** in the sofmax to control the diversity of generation, for example use $\tau=0.6$ to make it more variant than $\tau=1$.

Above all, you might find [this artical](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) great to help understand the task, where the author implemented a vanilla RNN language model to generate not only poems, but also linux kernel code. 

Requirements

1. Initialization. The embedding layer and the parameters for the model should be initialized before the training. Explain why you should not just initialize them to zero, and propose a way to initialize them properly. 10%
2. Generating Tang poem. Implement an LSTM to generate poems. Report the perplexity after your training, and generate poems that start with 日 、红、山、夜、湖、海、月, include the poem in your report. You are allowed to implement the LSTM with PyTorch but you should not relying on the LSTM cell or LSTM provided by it, just use its autograd module to help you deal with your gradient. Also you are not strictly required to follow everything described above, as long as you can make yourself clear in the report, you can use whatever setting that you believe is more appropriate for the task of generating poems. 50% 
   Bonus: you will earn up to 20% bonus (making the full mark 120% of this assignment) if you implement the gradient calculation and back propagation by yourself with numpy. Because you've implemented the LSTM in PyTorch you will have something to compare your gradient to, this will help you for gradient check. You might also use external data such as [全唐诗](<https://github.com/chinese-poetry/chinese-poetry>) to help your generation, this will be also be considered as bonus.
3. Optimization. We haven't mentioned how should you optimize your model, but of course you should use gradient descent. There are a lot of gradient descent algorithms that you could explore, a non inclusive name list: stochastic gradient descent, SGD with momentum [^2], Nesterov[^3], Adagrad[^4], Adadelta[^5], Adam[^6] etc. You should try at least two optimization algorithm to training your model. Note that as some of the algorithms require you to keep additional parameters across batches, **you should think about how it will influence the way you implement your gradient calculation** if you intended for the 20% bonus in the previous requirements. Include your comparison of the algorithms you've used in your report. 20%

[^Hochreiter & Schmidhuber (1997)]:http://www.bioinf.jku.at/publications/older/2604.pdf
[^ 2]: Qian, N. (1999). On the momentum term in gradient descent learning algorithms. Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151. <http://doi.org/10.1016/S0893-6080(98)00116-6>
[^ 3]: Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.
[^ 4]: Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from <http://jmlr.org/papers/v12/duchi11a.html>

[^ 5]: Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from <http://arxiv.org/abs/1212.5701>

[^ 6]:Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.