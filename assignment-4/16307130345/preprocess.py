import os
os.sys.path.append('..')
from sklearn.datasets import fetch_20newsgroups
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary

import string

target_name = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
               'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
               'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
               'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
               'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


def delete_char(doc):
  new_doc = ""
  for c in doc:
    if string.punctuation.find(c) == -1:
      new_doc += c
  return new_doc.lower().split()


def get_text_classification_datasets(num=10):
  categories = target_name[:num]
  train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
  test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
  train_data, train_target = [delete_char(doc) for doc in train.data], train.target.tolist()
  test_data, test_target = [delete_char(doc) for doc in test.data], test.target.tolist()

  # transform to DataSet()
  dataset_train, dataset_test = DataSet(), DataSet()
  max_len = 0
  for i in range(len(train_data)):
    dataset_train.append(Instance(doc_words=train_data[i], target=train_target[i]))
    if max_len < len(train_data[i]):
      max_len = len(train_data[i])
  for i in range(len(test_data)):
    dataset_test.append(Instance(doc_words=test_data[i], target=test_target[i]))
    if max_len < len(test_data[i]):
      max_len = len(test_data[i])

  # preprocess

  # drop some doc
  doc_len = lambda x: len(x['doc_words']) <= 10
  dataset_train.drop(doc_len)
  
  # build vocabulary
  vocab = Vocabulary(max_size=10000, min_freq=15, unknown='<unk>')
  dataset_train.apply(lambda x: [vocab.add(word) for word in x['doc_words']])
  vocab.build_vocab()

  # index
  indexF = lambda x: [vocab.to_index(word) for word in x['doc_words']]
  dataset_train.apply(indexF, new_field_name='words')
  dataset_test.apply(indexF, new_field_name='words')

  dataset_train_list = dataset_train.split(0.1)

  return dataset_train_list[0], dataset_train_list[1], dataset_test, len(vocab), max_len