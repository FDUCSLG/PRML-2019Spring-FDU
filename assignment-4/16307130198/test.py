import _pickle as pickle
import test_config
import sys
import os
import torch
import models
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import tqdm
import utils

import torch.nn.functional as F
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase

from fastNLP.core.optimizer import Optimizer
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler 
from fastNLP import Trainer
from fastNLP import Tester
from copy import deepcopy
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP.core import Adam
from fastNLP.core import SGD
from fastNLP.core.callback import EarlyStopCallback 

opt = test_config.Config()

def test():
    model_path = opt.model_path
    test_data = pickle.load(open(opt.test_data_path, 'rb'))
    
    vocab = pickle.load(open(opt.vocab, 'rb'))
    word2idx = vocab.word2idx
    idx2word = vocab.idx2word
    input_size = len(word2idx)
    
    vocab_size = opt.class_num
    class_num = opt.class_num 

    embedding_dim = opt.embedding_dim 
    
    if opt.model_name == "LSTMModel":
        model =  utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim)
    elif opt.model_name == "B_LSTMModel":
        model = utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim)
    elif opt.model_name == "CNNModel":
        model =  utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim)
    elif opt.model_name == "MyBertModel":
        test_data.apply(lambda x: x['input_data'][:2500], new_field_name='input_data')
        model = utils.find_class_by_name(opt.model_name, [models])(input_size=input_size,
                                                                   hidden_size=512,
                                                                   hidden_dropout_prob=0.1,
                                                                   num_labels = class_num,
                                                                  )

    utils.load_model(model, model_path)

    # define dataloader
    test_data.set_input('input_data', flag=True)
    test_data.set_target('target', flag=True)

    model_tester =Tester(data=test_data,
                         model=model,
                         batch_size=opt.batch_size,
                         device='cuda:1',
                         metrics=AccuracyMetric(pred="output", target="target"),
                        )
    model_tester.test()
    

if __name__ == '__main__':
    test()
    
