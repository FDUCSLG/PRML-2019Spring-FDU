import sys
import numpy as np

import torch
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP.core import Const
from fastNLP.models import CNNText
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Tester

import getopt
from cnn_model import CNN_model
from rnn_model import RNN_model
from train import cnn_train, rnn_train
from data import *

train_set, test_set, word_dict = get_dataset()

metrics = AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)
device = torch.device("cuda")


opts, args = getopt.getopt(sys.argv[1:], '', ['train', 'test', 'type='])

train_flag = False
tp = ''
test_flag = False
    
for name, val in opts:
    if name == '--train':
        train_flag = True
    if name == '--test':
        test_flag = True
    if name == '--type':
        tp = val

if tp == '' or (test_flag == False and train_flag == False):
    print('Args Invalid!')
    exit(0)        

if tp == 'rnn':
    model_rnn = RNN_model(dict_size = len(word_dict), embedding_dim = 128,
                          hidden_dim = 128, num_classes = 20)
    model_rnn.to(device)            
    if train_flag == True:
        rnn_train(epoch = 20, data = train_set, model = model_rnn)
    if test_flag == True:
        model_rnn.load_state_dict(torch.load('./rnn_state.pth'))
        tester = Tester(data = test_set, model = model_rnn, metrics = AccuracyMetric())
        tester.test()

if tp == 'cnn':
    model_cnn = CNN_model(dict_size = len(word_dict), embedding_dim = 128,
                          num_classes = 20, padding = 1, dropout = 0.1)
    model_cnn.to(device)            
    if train_flag == True:
        cnn_train(epoch = 20, data = train_set, model = model_cnn)
    if test_flag == True:
        model_cnn.load_state_dict(torch.load('./cnn_state.pth'))
        tester = Tester(data = test_set, model = model_cnn, metrics = AccuracyMetric())
        tester.test()
