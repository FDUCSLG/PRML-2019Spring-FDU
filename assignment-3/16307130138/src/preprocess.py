import sys
from sys import argv
import os
import numpy
import string
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester
from matplotlib import pyplot as plt
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('./PRML/assignment3')
from data.datacombine import get_all_data
from Configure import Config


def get_mini_data(data_path):
    data_set = DataSet()
    if os.path.exists(data_path):
        with open(data_path,'r',encoding='utf-8') as fin:
            sample = ""
            for lidx,line in enumerate(fin):
                line = line.strip()
                if(line == ""):
                    instance = Instance(raw_sentence=sample)
                    data_set.append(instance)
                    sample = ""
                else:
                    sample += line
    else:
        print("the data path doesn't  exit.")
    return data_set

def get_all_tang(data_path=None):
    data_set = DataSet()
    if data_path is None:
        all_tang=get_all_data()
        for sample in all_tang:
            instance = Instance(raw_sentence=sample)
            data_set.append(instance)
    else:
        if os.path.exists(data_path):
            with open(data_path,'r',encoding='utf-8') as fin:
                for lidx,line in enumerate(fin):
                    line = line.strip()
                    if(line != "" and len(line)>1):
                        instance = Instance(raw_sentence=line)
                        data_set.append(instance)
        else:
            print("the data path doesn't  exit.")
    return data_set


class PoemData(object):
    data_set = None
    train_data = None
    dev_data = None
    test_data = None
    vocab = None
    data_num = 0
    vocab_size = 0
    max_seq_len=0

    def split_sent(self,ins,remove_punc=False):
        line = ins['raw_sentence'].strip()
        words = ['<START>']
        for c in line:
            if c in ['，','。','？','！']:
                if remove_punc:
                    continue
                else:
                    words.append(c)
            else:
                words.append(c)
        words.append('<EOS>')
        self.max_seq_len = max(self.max_seq_len,len(words))
        return words

    def pad_seq(self,ins):
        words = ins['words']
        if(len(words) < self.max_seq_len):
            words = [0]*(self.max_seq_len-len(words)) + words
        else:
            words = words[:self.max_seq_len]
        return words

    def read_data(self,conf):
        if conf.all_tang:
            self.data_set = get_all_tang(conf.data_path)
        else: 
            self.data_set = get_mini_data(conf.data_path)
        self.data_num = len(self.data_set)
        self.data_set.apply(self.split_sent,new_field_name='words')
        self.max_seq_len = min(self.max_seq_len,conf.max_seq_len)
        self.data_set.apply(lambda x : len(x['words']),new_field_name='seq_len')
        self.train_data,self.test_data = self.data_set.split(0.2)
        
    def get_vocab(self):
        self.vocab = Vocabulary(min_freq=1)
        self.train_data.apply(lambda x : [self.vocab.add(word) for word in x['words']])
        self.vocab.build_vocab()
        self.vocab.build_reverse_vocab()
        self.vocab_size = self.vocab.__len__()

        self.train_data.apply(lambda x : [self.vocab.to_index(word) for word in x['words']],new_field_name='words')
        self.train_data.apply(self.pad_seq,new_field_name='pad_words')
        
        self.test_data.apply(lambda x : [self.vocab.to_index(word) for word in x['words']],new_field_name='words')
        self.test_data.apply(self.pad_seq,new_field_name='pad_words')

        print(self.test_data[0])
    
    def get_data(self,conf):
        self.get_data(conf)
        self.get_vocab()
        return self.train_data['pad_words'], self.test_data['pad_words']


if __name__ == "__main__":
    conf = Config()
    data = PoemData()
    data.read_data(conf)
    print(len(data.data_set))
    print(len(data.train_data))
    data.get_vocab()

