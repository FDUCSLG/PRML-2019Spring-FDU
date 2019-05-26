## <center>Pattern Recognition and Machine Learning</center>

### <center>Fudan University / 2019 Spring</center>

<center>Assignment 4</center>

In this assignment you are going to re-do the text classification task in assignment 2 with FastNLP and PyTroch with more powerful tools, RNN and CNN.

#### Description

Familiar as you are with the text classification task as you've already finished it with logistic regression in assignment 2, the challenge of this task is to use RNN and CNN to achieve the same goal. As long as you finish this task with FastNLP, you are welcomed to use any recent developed architecture to improve the performance (note that it won't not neccessarily become better than logistic regression, since we have a rather small dataset)

Here are some references that you might find helpful

* Word2Vec: Distributed Representations of Words and Phrases and their Compositionality [link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
* GloVe: Global Vectors for Word Representation [link](https://nlp.stanford.edu/projects/glove/)
* CNN: Convolutional Neural Networks for Sentence Classification [link](https://www.aclweb.org/anthology/D14-1181)
* CNN: Character-level Convolutional Networks for Text Classification [link](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
* RNN: Recurrent Convolutional Neural Networks for Text Classification [link](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
* RNN: Recurrent Neural Network for Text Classification with Multi-Task Learning,  [link](https://arxiv.org/pdf/1605.05101.pdf)

Requirements:

* Implement two text classifiers in RNN and CNN to classifiy the dataset provided in assignment 2 with FastNLP framework, including its data processing module and training support. Report your methods and your result. You could use a larger portion of the dataset and it will be considered as a bonus, you might find these link useful [The 20 newsgroups text dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), you could also additionally use  other dataset to have more interesting result. 70%
* Write your thoughts about FastNLP, including your experience of using it and how could it be improved, you might also do a survey about other NLP frameworks built on top of deep learning to make your ideas more solid. 30%