import os
os.sys.path.append('..')
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.models import CNNText
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Tester
from fastNLP import Callback

import torch
import torch.nn as nn

from preprocess import get_text_classification_datasets
from model import TextCNN, TextRNN

class_num = 10
dataset_train, dataset_dev, dataset_test, vocab_size, max_test_len = get_text_classification_datasets(class_num)


class TextCNNConfig(object):
  num_class = class_num
  vocab_s = vocab_size
  embedding_s = 50
  feat_s = 100
  window_s = [3, 4, 5]
  max_len = max_test_len
  dropout_rate = 0.5

class TextRNNConfig(object):
  vocab_s = vocab_size
  embedding_s = 128
  hidden_s = 128
  dropout_rate = 0.5
  num_class = class_num
  max_len = max_test_len


# dataset config
dataset_train.set_input('words')
dataset_train.set_target('target')
dataset_dev.set_input('words')
dataset_dev.set_target('target')
dataset_test.set_input('words')
dataset_test.set_target('target')


def train_TextCNN():
  model = TextCNN(TextCNNConfig)
  loss = CrossEntropyLoss(pred="pred", target="target")
  metrics = AccuracyMetric(pred="pred", target="target")
  trainer = Trainer(model=model,
                    train_data=dataset_train,
                    dev_data=dataset_dev,
                    loss=loss,
                    metrics=metrics,
                    batch_size=16,
                    n_epochs=15)
  trainer.train()
  tester = Tester(dataset_test, model, metrics)
  tester.test()


def train_TextRNN():
  model = TextRNN(TextRNNConfig)
  loss = CrossEntropyLoss(pred="pred", target="target")
  metrics = AccuracyMetric(pred="pred", target="target") 
  trainer = Trainer(model=model,
                    train_data=dataset_train,
                    dev_data=dataset_dev,
                    loss=loss,
                    metrics=metrics,
                    batch_size=16,
                    n_epochs=20)
  trainer.train()
  tester = Tester(dataset_test, model, metrics)
  tester.test()



# train_TextCNN()
train_TextRNN()










