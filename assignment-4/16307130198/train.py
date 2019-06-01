import _pickle as pickle
import config
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
from copy import deepcopy
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP.core import Adam
from fastNLP.core import SGD
from fastNLP.core.callback import EarlyStopCallback 

opt = config.Config()

def train():
    train_data = pickle.load(open(opt.train_data_path, 'rb'))
    validate_data = pickle.load(open(opt.validate_data_path, 'rb'))
    
    vocab = pickle.load(open(opt.vocab, 'rb'))
    word2idx = vocab.word2idx
    idx2word = vocab.idx2word
    input_size = len(word2idx)
    
    vocab_size = opt.class_num
    class_num = opt.class_num

    embedding_dim = opt.embedding_dim 
    
    
    if opt.model_name == "LSTMModel":
        model =  utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim, opt.use_word2vec, opt.embedding_weight_path)
    elif opt.model_name == "B_LSTMModel":
        model = utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim, opt.use_word2vec, opt.embedding_weight_path)
    elif opt.model_name == "CNNModel":
        model =  utils.find_class_by_name(opt.model_name, [models])(input_size, vocab_size, embedding_dim, opt.use_word2vec, opt.embedding_weight_path)
    elif opt.model_name == "MyBertModel":
        #bert_dir = "./BertPretrain"
        #bert_dir = None
        #model = utils.find_class_by_name(opt.model_name, [models])(10, 0.1, 4, bert_dir)
        train_data.apply(lambda x: x['input_data'][:2500], new_field_name='input_data')
        validate_data.apply(lambda x: x['input_data'][:2500], new_field_name='input_data')
        
        model = utils.find_class_by_name(opt.model_name, [models])(input_size=input_size,
                                                                   hidden_size=512,
                                                                   hidden_dropout_prob=0.1,
                                                                   num_labels = class_num,
                                                                   use_word2vec=opt.use_word2vec,
                                                                   embedding_weight_path=opt.embedding_weight_path,
                                                                  )

    if not os.path.exists(opt.save_model_path):
        os.mkdir(opt.save_model_path)
    
    # define dataloader
    train_data.set_input('input_data', flag=True)
    train_data.set_target('target', flag=True)
    validate_data.set_input('input_data', flag=True)
    validate_data.set_target('target', flag=True)
    
    if opt.optimizer == 'SGD':
        _optimizer = SGD(lr=opt.learning_rate, momentum=0)
    elif opt.optimizer == 'SGD_momentum':
        _optimizer = SGD(lr=opt.learning_rate, momentum=0.9)
    elif opt.optimizer == 'Adam':
        _optimizer=Adam(lr=opt.learning_rate, weight_decay=0)

    overfit_trainer =Trainer(model=model,
                             train_data = train_data,
                             loss=CrossEntropyLoss(pred="output", target="target"),
                             n_epochs=opt.epoch,
                             batch_size=opt.batch_size,
                             device=[0,1,2,3],
                             #device=None,
                             dev_data=validate_data,
                             metrics=AccuracyMetric(pred="output", target="target"),
                             metric_key="+acc",
                             validate_every=opt.validate_every,
                             optimizer = _optimizer,
                             callbacks=[EarlyStopCallback(opt.patience)],
                             save_path=opt.save_model_path)

    overfit_trainer.train()
    

if __name__ == '__main__':
    train()
    
