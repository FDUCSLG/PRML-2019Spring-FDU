import os
os.sys.path.append('..')
from sklearn.datasets import fetch_20newsgroups
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.models import CNNText
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

import torch
import torch.nn as nn

import string

# =================== utils function =====================

def delete_char(doc):
  new_doc = ""
  for c in doc:
    if string.punctuation.find(c) == -1:
      new_doc += c
  return " ".join(new_doc.lower().split())


def get_text_classification_datasets():
  dataset_train, dataset_test = DataSet(), DataSet()
  train = fetch_20newsgroups(subset='train', data_home='../../..')
  test = fetch_20newsgroups(subset='test', data_home='../../..')
  train_data, train_target = [delete_char(doc) for doc in train.data], train.target.tolist()
  test_data, test_target = [delete_char(doc) for doc in test.data], test.target.tolist()
  for i in range(len(train_data)):
    dataset_train.append(Instance(doc=train_data[i], target=train_target[i]))
  for i in range(len(test_data)):
    dataset_test.append(Instance(doc=test_data[i], target=test_target[i]))
  return dataset_train, dataset_test


dataset_train, dataset_test = get_text_classification_datasets()

print("loadding data finished!!")

# doc split
splitF = lambda ins: ins['doc'].split()
dataset_train.apply(splitF, new_field_name='words')
dataset_test.apply(splitF, new_field_name='words')

# drop some doc
doc_len = lambda x: len(x['words']) <= 10
dataset_train.drop(doc_len)
dataset_test.drop(doc_len)

# build vocabulary
vocab = Vocabulary(max_size=10000, min_freq=20, unknown='<unk>', padding='<pad>')
dataset_train.apply(lambda x: [vocab.add(word) for word in x['words']])
vocab.build_vocab()

print("build vocabulary finished!!")

# index
indexF = lambda x: [vocab.to_index(word) for word in x['words']]
dataset_train.apply(indexF, new_field_name='word_seq')
dataset_test.apply(indexF, new_field_name='word_seq')

# add doc length
doc_len = lambda x: len(x['words'])
dataset_train.apply(doc_len, new_field_name='seq_len')

# dataset config
dataset_train.set_input('word_seq')
dataset_train.set_target('target')
dataset_test.set_input('word_seq')
dataset_test.set_target('target')

print("begin training!!!")


model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=20, padding=2, dropout=0.1)
loss = CrossEntropyLoss(pred="output", target="target")
metrics = AccuracyMetric(pred="predict", target="target")
trainer = Trainer(model=model,
                  train_data=dataset_train,
                  dev_data=dataset_test,
                  loss=loss,
                  metrics=metrics,
                  batch_size=16,
                  n_epochs=10,
                  save_path=None)

trainer.train()

print("train finished!!")