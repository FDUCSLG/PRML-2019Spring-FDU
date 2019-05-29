import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.optimizer import Adam,SGD
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core.callback import EarlyStopCallback
from torch.autograd import Variable
from model import CNN,RNN
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
import pickle
import config

if __name__=="__main__":

    vocab = pickle.load(open(config.vocab_path, 'rb'))
    train_data = pickle.load(open(config.train_data_path, 'rb'))
    dev_data = pickle.load(open(config.dev_data_path, 'rb'))
    test_data = pickle.load(open(config.test_data_path, 'rb'))

    if config.model == "CNN":
        model = CNN(len(vocab), config.intput_size, config.class_num)
    elif config.model == "RNN":
        model = RNN(len(vocab), config.intput_size, config.hidden_size, config.class_num, config.rnn_type)

    optimizer=Adam(lr=config.learning_rate, weight_decay=0)
    loss = CrossEntropyLoss(pred="output", target="target")
    metrics = AccuracyMetric(pred="output", target="target")
    trainer = Trainer(  model=model,
                        n_epochs=config.epoch,
                        validate_every=config.validate_every, 
                        optimizer=optimizer, 
                        train_data=train_data, 
                        dev_data=dev_data, 
                        metrics=metrics,
                        loss=loss, 
                        batch_size=config.batch_size, 
                        device='cuda:0',
                        save_path=config.save_path,
                        metric_key="acc",
                        callbacks=[EarlyStopCallback(config.patience)])
    trainer.train()