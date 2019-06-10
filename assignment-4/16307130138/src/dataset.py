import os
os.sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import string
import time
import math

import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const

from sklearn.datasets import fetch_20newsgroups
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

def get_text_classification_datasets():
    categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
    dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../')
    dataset_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test

def get_all_20news():
    dataset_train = fetch_20newsgroups(subset='train', data_home='../../',remove=('headers'))
    dataset_test = fetch_20newsgroups(subset='test', data_home='../../',remove=('headers'))
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    print("The header of data is removed.")
    return dataset_train, dataset_test

def combine_whitespace(s):
    return s.split()

def tokenize(data_train,data_test,max_num=100000):
    train = []
    test = []
    for i in range(len(data_train)):
        text = data_train[i]
        if text=="":
            print("Empty text.{0}".format(i))
        elif text is None:
            print("None type text.")
        newtext=""
        for c in text:
            if c not in string.punctuation:
                newtext += c
            else:
                newtext += ' '
        newtext = combine_whitespace(newtext.lower())
        # data_train[i] = newtext
        if newtext is None:
            train.append(newtext)
    
    for i in range(len(data_test)):
        text = data_test[i]
        if text=="":
            print("Empty text.{0}".format(i))
        elif text is None:
            print("None type text.")
        newtext=""
        for c in text:
            if c not in string.punctuation:
                newtext += c
            else:
                newtext += ' '
        newtext = combine_whitespace(newtext.lower())
        #data_test[i] = newtext
        if newtext is not None:
            test.append(newtext)
    # return data_train,data_test
    return train,test

def text2multi_hot(words,vocab_size,word2index=None):
    multi_hot_vector = [0]*(vocab_size)
    for word in words:
        multi_hot_vector[word] = 1
    return multi_hot_vector

def class2target(class_type,class_num):
    target = [0]*class_num
    target[class_type] = 1
    return target

class TextData():
    vocab_size = 0
    dataset_size = 0
    train_size = 0
    test_size = 0
    class_num = 4
    min_count = 10
    max_seq_len = 500
    seq_limit = 2000
    data_src = "20news"

    data_set = DataSet()
    train_set = DataSet()
    test_set = DataSet()
    dev_set = DataSet()
    vocab = None


    def __init__(self,data_src="20news",min_count=10,seq_limit=None):
        self.data_src = data_src
        self.min_count = min_count
        if seq_limit is not None:
            self.seq_limit = seq_limit

    def find_max_len(self,words):
        self.max_seq_len = max(len(words),self.max_seq_len)

    def seq_regularize(self,words):
        wlen = len(words)
        if wlen<self.max_seq_len:
            return [0]*(self.max_seq_len-wlen) + words
        else:
            return words[:self.max_seq_len]

    def fetch_20news(self,size=4):
        print("Loading 20newsgroups data and tokenize.")
        if size==20:
            train,test = get_all_20news()
        else:
            train,test = get_text_classification_datasets()
        train_input,test_input = tokenize(train.data,test.data)
        train_target = train.target
        test_target = test.target
        self.class_num = len(train.target_names)
        assert (self.class_num == len(test.target_names))

        # Building Fastnlp dataset.
        print("Building Fastnlp dataset.")
        self.train_set = DataSet({"text":train_input,"class":train_target})
        self.test_set = DataSet({"text":test_input,"class":test_target})
        
        # Building Fastnlp vocabulary...
        print("Building Fastnlp vocabulary.")
        self.vocab = Vocabulary(min_freq=self.min_count)
        self.train_set.apply(lambda x : [self.vocab.add_word(word) for word in x['text']])
        self.vocab.build_vocab()
        self.vocab.build_reverse_vocab()
        self.vocab_size = len(self.vocab)
        # Building multi-hot-vector for train_set and test_set.
        print("Building id-presentation for train_set and test_set.")
        self.vocab.index_dataset(self.train_set,self.test_set,field_name='text',new_field_name='words')
        
        self.train_set.apply_field(lambda x : len(x),field_name='words',new_field_name='seq_len')
        self.test_set.apply_field(lambda x : len(x),field_name='words',new_field_name='seq_len')
        self.train_set.apply_field(self.find_max_len,field_name='words')

        print(self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len,self.seq_limit)

        self.train_set.apply_field(self.seq_regularize,field_name='words',new_field_name='words')
        self.test_set.apply_field(self.seq_regularize,field_name='words',new_field_name='words')
        # self.train_set.apply(lambda x : text2multi_hot(x['words'],self.vocab_size),new_field_name="input")
        # self.test_set.apply(lambda x : text2multi_hot(x['words'],self.vocab_size),new_field_name='input')
        
        # Building target-vector for train_set and test_set.
        print("Building target-vector for train_set and test_set.")
        self.train_set.apply(lambda x : int(x['class']),new_field_name="target",is_target=True)
        self.test_set.apply(lambda x : int(x['class']),new_field_name="target",is_target=True)
        # self.train_set.apply(lambda x : class2target(x['class'],self.calss_num),new_field_name="target")
        # self.test_set.apply(lambda x : class2target(x['class'],self.calss_num),new_field_name="target")

    def fetch_csv(self,path=None):
        print("Not implemented now...")
        pass

    def fetch_data(self,path=None):
        if self.data_src == "20news":
            # Loading 20newsgroups data and tokenize.
            self.fetch_20news()
        elif self.data_src == "20news_all":
            self.fetch_20news(size=20)
        else:
            print("No data src...")
        
        self.train_size = self.train_set.get_length()
        self.test_size = self.test_set.get_length()
        return self.train_size,self.test_size


if __name__ == "__main__":
    data = TextData(data_src='20news')
    print(data.fetch_data())
    len_lst = data.train_set.get_field('seq_len')
    plt.hist(len_lst,bins=500)
    #plt.show()
    plt.savefig()
    print("Test done.")
    









