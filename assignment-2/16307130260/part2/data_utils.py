import os
os.sys.path.append('../..')
from handout import get_text_classification_datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import re
import string
import numpy as np
from matplotlib import pyplot as plt


class data_processor:
    def __init__(self, min_count=10):
        self.min_count = min_count

    # Preprocess the data into bag of words. 
    # return:
    # data  Multi-hot
    # class  One-hot
    # categories  Name of class in one-hot array
    def generate_vocabulary(self, data):
        data_splited = self.split_word(data)
        self.vocabulary = self.get_vocabulary(data_splited)
        return len(self.vocabulary.keys())

    def process_data(self, data, C, num_classes):
        data_splited = self.split_word(data)
        multi_hot = self.get_multi_hot(data_splited)
        
        one_hot = self.get_one_hot(C, num_classes)

        return multi_hot, one_hot

    def split_word(self, data):
        data_ign_pun = [ re.sub(r'[{}]+'.format(string.punctuation), "", d.lower()) for d in data ]
        data_cleaned = [ re.sub(r'[{}]+'.format(string.whitespace), " ", d) for d in data_ign_pun]
        data_splited = [ d.split(" ") for d in data_cleaned]
        return data_splited
    
    def get_vocabulary(self, data_splited):
        # count for all words
        sum_dic = {}
        for item in data_splited:
            for s in item:
                if s in sum_dic.keys():
                    sum_dic[s] += 1
                else:
                    sum_dic[s] = 1
        # get dict of vocabulary 
        vocabulary = {}
        cnt = 0
        for key, value in sum_dic.items():
            if value >= self.min_count:
                vocabulary[key] = cnt
                cnt += 1
        return vocabulary

    def get_multi_hot(self, data_splited):
        words = self.vocabulary.keys()

        len_of_data = len(data_splited)
        len_of_voc = len(words)

        result = np.zeros((len_of_data, len_of_voc))
        for i in range(len_of_data):
            for s in data_splited[i]:
                if s in words:
                    result[i][self.vocabulary[s]] = 1
        return result

    def get_one_hot(self, C, num_classes):
        cl = len(C)
        one_hot = np.zeros((cl, num_classes))
        one_hot[np.arange(cl),C] = 1
        return one_hot

if __name__ == "__main__":
    training_set, test_set = get_text_classification_datasets()
    dp = data_processor()
    dp.generate_vocabulary(training_set.data)
    x, y = dp.process_data(training_set.data, training_set.target, len(training_set.target_names))
    print(x.shape)
    print(y.shape)
    x, y = dp.process_data(test_set.data, test_set.target, len(test_set.target_names))
    print(x.shape)
    print(y.shape)

