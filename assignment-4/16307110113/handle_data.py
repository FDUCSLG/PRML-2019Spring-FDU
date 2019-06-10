# -*- coding: utf-8 -*-
import os
os.sys.path.append('../../assignment-2/handout')

import math
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import re
from sklearn.datasets import fetch_20newsgroups
from __init__ import *

#超参数
text_len = 400
min_freqency = 10

def get_all_text_classification_datasets():
    dataset_train = fetch_20newsgroups(subset='train', data_home='../../..')
    dataset_test = fetch_20newsgroups(subset='test', data_home='../../..')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test


def preprocessing(data_train, data_test):
    data_train_dict = {'raw_text': data_train.data,
                        'label': data_train.target}
    data_test_dict = {'raw_text': data_test.data,
                        'label': data_test.target}
    dataset = DataSet(data_train_dict)
    test_set = DataSet(data_test_dict)
    dataset.apply_field(lambda piece: re.sub('[' + string.whitespace + '\u200b]+', ' ', 
                        re.sub('[' + string.punctuation +']', '', piece)).strip().lower(), 
                        field_name='raw_text', new_field_name='raw_text')
    test_set.apply_field(lambda piece: re.sub('[' + string.whitespace + '\u200b]+', ' ', 
                        re.sub('[' + string.punctuation + ']', '', piece)).strip().lower(), 
                        field_name='raw_text', new_field_name='raw_text')
    dataset.apply_field(lambda piece: piece.split(' '), 
                        field_name='raw_text', new_field_name='text')
    test_set.apply_field(lambda piece: piece.split(' '), 
                        field_name='raw_text', new_field_name='text')

    # 观察数据集中文本长度分布，以选取合适的text_length
    # data_lens = []
    # for instance in dataset:
    #     data_lens.append(len(instance['text']))
    # for instance in test_set:
    #     data_lens.append(len(instance['text']))
    # print("max text_len %d, min text_len %d" % (max(data_lens), min(data_lens)))
    # print(len([i for i in data_lens if i < 400]))
    # plt.hist(data_lens, bins=200, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("text_length")
    # plt.ylabel("number of texts")
    # plt.title("Distribution of text_length")
    # plt.show()

    dataset.apply_field(lambda piece: piece[:text_len], 
                        field_name='text', new_field_name='text')
    test_set.apply_field(lambda piece: piece[:text_len], 
                        field_name='text', new_field_name='text')
    
    dataset.delete_field('raw_text')
    test_set.delete_field('raw_text')

    # 将数字都转换成相同的形式
    # for instance in dataset:
    #     for i, word in enumerate(instance['text']):
    #         if word.isdigit():
    #             instance['text'][i] = '1'
    # for instance in test_set:
    #     for i, word in enumerate(instance['text']):
    #         if word.isdigit():
    #             instance['text'][i] = '1'

    vocab = Vocabulary(min_freq=min_freqency, unknown='<unk>', padding='<pad>').from_dataset(dataset, field_name='text')
    print("vocabulary_length:", len(vocab))
    vocab.index_dataset(dataset, field_name='text',new_field_name='text')
    vocab.index_dataset(test_set, field_name='text',new_field_name='text')

    # 是否使用padding, 将每条文本变为等长

    train_set, dev_set = dataset.split(0.2)

    train_set.rename_field('text', Const.INPUT)
    train_set.rename_field('label', Const.TARGET)
    train_set.set_input(Const.INPUT)
    train_set.set_target(Const.TARGET)
    dev_set.rename_field('text', Const.INPUT)
    dev_set.rename_field('label', Const.TARGET)
    dev_set.set_input(Const.INPUT)
    dev_set.set_target(Const.TARGET)
    test_set.rename_field('text', Const.INPUT)
    test_set.rename_field('label', Const.TARGET)
    test_set.set_input(Const.INPUT)
    test_set.set_target(Const.TARGET)

    print("train_set length:", len(train_set))
    print("dev_set length:", len(dev_set))
    print("test_set length:", len(test_set))

    return train_set, dev_set, test_set, vocab


def save_vocab(file_name, vocab):
    with open(file_name,'w') as f:
        for i in range(len(vocab)):
            word = vocab.to_word(i)
            if word != '<unk>' and word != '<pad>':
                f.writelines(word+"\n")


if __name__ == "__main__":
    print("1: handle small dataset\n2: handle larger dataset\nInput 1 or 2: ")
    choice = input()
    if choice == "1":
        rawdata_train, rawdata_test = get_text_classification_datasets()
    elif choice == "2":
        rawdata_train, rawdata_test = get_all_text_classification_datasets()

    # print(rawdata_test.data[0])
    # print(rawdata_test.target[0])
    train_set, dev_set, test_set, vocab = preprocessing(rawdata_train, rawdata_test)
    save_vocab("vocab.txt", vocab)
    train_set.save("train_set")
    dev_set.save("dev_set")
    test_set.save("test_set")