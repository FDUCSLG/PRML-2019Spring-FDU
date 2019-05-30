import os
import sys
sys.path.append('..')
import string
import pickle
import re
from sklearn.datasets import fetch_20newsgroups, load_files
from fastNLP import Vocabulary, DataSet, Instance

def get_20newsgroups_data(categories):
  dataset_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories, data_home='../data/') # , download_if_missing=False)
  dataset_test = fetch_20newsgroups(subset='test', shuffle=True, categories=categories, data_home='../data/') # , download_if_missing=False)
  print("In training dataset:")
  print('Samples:', len(dataset_train.data))
  print('Categories:', len(dataset_train.target_names))
  print("In testing dataset:")
  print('Samples:', len(dataset_test.data))
  print('Categories:', len(dataset_test.target_names))
  return dataset_train, dataset_test

def sentence_to_words(sentence):
  regular = re.compile(r'[\s]+')
  words = regular.split( sentence.translate( str.maketrans('', '', string.punctuation) ).lower() )
  return words

def create_dataset(data, sample_size):
  data_set = DataSet()
  data_set.add_field('raw_sentence', data.data[:sample_size])
  data_set.add_field('target', data.target[:sample_size])
  data_set.apply(lambda x: sentence_to_words( x['raw_sentence'] ), new_field_name='word_seq')
  return data_set

def build_dataset(train_size, test_rate, categories):
  vocab = load('../data/vocab')
  train_set = load('../data/train_set')
  test_set = load('../data/test_set')
  if not vocab is None and not train_set is None and not test_set is None:
    return vocab, train_set, test_set
  
  train, test = get_20newsgroups_data(categories)
  train_set = create_dataset(train, train_size)
  test_set = create_dataset(test, int(train_size * test_rate))
  # vocabulary
  vocab = Vocabulary(min_freq=10)
  test_set.apply(lambda x: [vocab.add(word) for word in x['word_seq']])
  vocab.build_vocab()
  # word_seq to int
  train_set.apply(lambda x: [vocab.to_index(word) for word in x['word_seq']], new_field_name='word_seq')
  test_set.apply(lambda x: [vocab.to_index(word) for word in x['word_seq']], new_field_name='word_seq')
  # tag
  train_set.set_input('word_seq')
  train_set.set_target('target')
  test_set.set_input('word_seq')
  test_set.set_target('target')

  save('../data/vocab', vocab)
  save('../data/train_set', train_set)
  save('../data/test_set', test_set)
  return vocab, train_set, test_set

def load(filename):
  if os.path.exists(filename):
    with open (filename, 'rb') as fp:
      return pickle.load(fp)
  else:
    return None

def save(filename, item):
  with open(filename, 'wb') as fp:
    pickle.dump(item, fp)