import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.optimizer import Adam,SGD
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
from fastNLP.core.callback import EarlyStopCallback
from torch.autograd import Variable
from model import PoetryModel
from data import Init_Dataloader,Get_Data_Vocab
from generate import generate
import torch.utils.data as data
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
import pickle
import config

class MyCrossEntropyLoss(LossBase):
    def __init__(self, pred=None, target=None, padding_idx=-100):
        super(MyCrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = padding_idx

    def get_loss(self, pred, target):
        loss = F.cross_entropy(input=pred, target=target.view(-1),
                               ignore_index=self.padding_idx)
        print(loss)
        return loss

class PerplexityMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super(PerplexityMetric, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.count = 0
        self.PP_sum = 0
    
    def evaluate(self, pred, target):
        batch, seq_len = target.shape
        pred = pred.float()
        target = target.view(-1,1).long()
        pred = F.softmax(pred, dim=1)

        target_onehot = torch.zeros_like(pred)
        target_onehot = target_onehot.scatter(1, target, 1)

        x = torch.sum(torch.mul(pred, target_onehot), dim=1).view(batch, seq_len)
        self.count += batch
        self.PP_sum += torch.sum(torch.exp(torch.mean(-torch.log(x), dim=1)))
 
    def get_metric(self, reset=True):
        evaluate_result = {'PPL':self.PP_sum/self.count}
        if reset:
            self.count = 0
            self.PP_sum = 0
        return evaluate_result

class Adagrad(Optimizer):
    def __init__(self, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adagrad, self).__init__(model_params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
    
    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adagrad(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adagrad(self._get_require_grads_param(self.model_params), **self.settings)

if __name__=="__main__":

    vocab = pickle.load(open(config.vocab_path, 'rb'))
    train_data = pickle.load(open(config.train_data_path, 'rb'))
    dev_data = pickle.load(open(config.dev_data_path, 'rb'))

    model = PoetryModel(len(vocab), config.intput_size, config.hidden_size)
    optimizer=Adam(lr=config.learning_rate, weight_decay=0)
    # optimizer = Adagrad(lr=config.learning_rate, weight_decay=0)
    # optimizer=SGD(lr=config.learning_rate, momentum=0.9)
    loss = MyCrossEntropyLoss(pred="output", target="target")
    metric = PerplexityMetric(pred="output", target="target")
    trainer = Trainer(  model=model,
                        n_epochs=config.epoch,
                        validate_every=config.validate_every, 
                        optimizer=optimizer, 
                        train_data=train_data, 
                        dev_data=dev_data, 
                        metrics=metric,
                        loss=loss, 
                        batch_size=config.batch_size, 
                        device='cuda:0',
                        save_path=config.save_path,
                        metric_key="-PPL",
                        callbacks=[EarlyStopCallback(config.patience)])
    trainer.train()