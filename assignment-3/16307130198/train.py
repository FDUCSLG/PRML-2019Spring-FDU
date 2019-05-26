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

class MyCrossEntropyLoss(LossBase):
    def __init__(self, pred=None, target=None, padding_idx=0):
        super(MyCrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = padding_idx
        
    def get_loss(self, pred, target):
        _loss =  F.cross_entropy(input=pred, target=target.view(-1),
                               ignore_index=self.padding_idx)
        return _loss 

class MyPPMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_lens=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_lens=seq_lens)
        self.total = 0
        self.pp = 0

    def evaluate(self, pred, target, seq_lens=None):
        batch_size, seq_len = target.shape
        real_seq_len = torch.sum(target!=0, dim=1).float()

        target = target.view(-1,1).long()
        
        pred = pred.view(batch_size*seq_len, -1)
        pred = F.softmax(pred, dim=1)

        target_onehot = torch.zeros_like(pred)
        target_onehot = target_onehot.scatter(1, target, 1)
        
        # cancel the influence of pad
        pred[:, 0] = 1.0
        x = torch.sum(torch.mul(pred, target_onehot), dim=1).view(batch_size, seq_len)
        self.pp += torch.sum(torch.exp(torch.sum(-torch.log(x),dim=1)/real_seq_len ))
        
        self.total += batch_size 
    
    def get_metric(self, reset=True):
        metric = {}
        metric['pp'] = self.pp / self.total
        if reset:
            self.pp = 0
            self.total = 0
        return metric

class Adagrad(Optimizer):
    def __init__(self, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adagrad, self).__init__(model_params, lr=lr, lr_decay=lr_decay, 
                                      weight_decay=weight_decay,
                                      initial_accumulator_value=initial_accumulator_value)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            return torch.optim.Adagrad(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adagrad(self._get_require_grads_param(self.model_params), **self.settings)

opt = config.Config()


def train():
    train_data = pickle.load(open(opt.train_data_path, 'rb'))
    validate_data = pickle.load(open(opt.validate_data_path, 'rb'))
    
    vocab = pickle.load(open(opt.vocab, 'rb'))
    word2idx = vocab.word2idx
    idx2word = vocab.idx2word
    vocab_size = len(word2idx)
    print("vocab_size" + str(vocab_size))
    embedding_dim = opt.embedding_dim 
    hidden_dim = opt.hidden_dim
    model =  utils.find_class_by_name(opt.model_name, [models])(vocab_size, embedding_dim, hidden_dim)
    
    if not os.path.exists(opt.save_model_path):
        os.mkdir(opt.save_model_path)

    # define dataloader
    train_data.set_input('input_data', flag=True)
    train_data.set_target('target', flag=True)
    validate_data.set_input('input_data', flag=True)
    validate_data.set_target('target', flag=True)
    
    if opt.optimizer == 'Adagrad':
        _optimizer = Adagrad(lr=opt.learning_rate, weight_decay=0)
    elif opt.optimizer == 'SGD':
        _optimizer = SGD(lr=opt.learning_rate, momentum=0)
    elif opt.optimizer == 'SGD_momentum':
        _optimizer = SGD(lr=opt.learning_rate, momentum=0.9)
    elif opt.optimizer == 'Adam':
        _optimizer=Adam(lr=opt.learning_rate, weight_decay=0)

    overfit_trainer =Trainer(model=model,
                             train_data = train_data,
                             loss=MyCrossEntropyLoss(pred="output", target="target"),
                             n_epochs=opt.epoch,
                             batch_size=opt.batch_size,
                             device='cuda:0',
                             dev_data=validate_data,
                             metrics=MyPPMetric(pred="output", target="target"),
                             metric_key="-pp",
                             validate_every=opt.validate_every,
                             optimizer = _optimizer,
                             callbacks=[EarlyStopCallback(opt.patience)],
                             save_path=opt.save_model_path)

    overfit_trainer.train()
    

if __name__ == '__main__':
    train()
    
