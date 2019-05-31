import numpy as np
import re
import string
import pickle

from sklearn.datasets import fetch_20newsgroups
from fastNLP import DataSet
from fastNLP import Vocabulary
from fastNLP import Const

def read_data():
    with open("data.bin", "rb") as f:
        return pickle.load(f)

def get_text_classification_datasets():
    categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
    dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
    dataset_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
    print("******************************")
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    print("******************************")
    return dataset_train, dataset_test


def get_data():
    dataset_train, dataset_test = get_text_classification_datasets()
    # print(dataset_train.data)

    dic_train = {
        "input" : dataset_train.data,
        "target" : dataset_train.target
    }
    dic_test = {
        "input" : dataset_test.data,
        "target" : dataset_test.target
    }

    dataset = DataSet(dic_train)
    test_data = DataSet(dic_test)

    dataset.apply_field(lambda x: re.sub(r'[{}]+'.format(string.punctuation), "", x.lower()), field_name='input', new_field_name='input')
    dataset.apply_field(lambda x: re.sub(r'[{}]+'.format(string.whitespace), " ", x), field_name='input', new_field_name='input')
    dataset.apply_field(lambda x: x.split(), field_name='input', new_field_name='words')

    test_data.apply_field(lambda x: re.sub(r'[{}]+'.format(string.punctuation), "", x.lower()), field_name='input', new_field_name='input')
    test_data.apply_field(lambda x: re.sub(r'[{}]+'.format(string.whitespace), " ", x), field_name='input', new_field_name='input')
    test_data.apply_field(lambda x: x.split(), field_name='input', new_field_name='words')


    # **************************
    dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    test_data.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    dataset.rename_field('words', Const.INPUT)
    dataset.rename_field('seq_len', Const.INPUT_LEN)
    dataset.rename_field('target', Const.TARGET)
    
    test_data.rename_field('words', Const.INPUT)
    test_data.rename_field('seq_len', Const.INPUT_LEN)
    test_data.rename_field('target', Const.TARGET)

    # dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    dataset.set_input(Const.INPUT)
    dataset.set_target(Const.TARGET)

    # test_data.set_input(Const.INPUT, Const.INPUT_LEN)
    test_data.set_input(Const.INPUT)
    test_data.set_target(Const.TARGET)
    # **************************

    # only use train for vocab or train+dev
    train_data, dev_data = dataset.split(0.1)
    # print(len(train_data), len(dev_data), len(test_data))
    # print(train_data[0])

    vocab = Vocabulary(min_freq=10).from_dataset(train_data, field_name=Const.INPUT)

    vocab.index_dataset(train_data, field_name=Const.INPUT,new_field_name=Const.INPUT)
    vocab.index_dataset(dev_data, field_name=Const.INPUT,new_field_name=Const.INPUT)
    vocab.index_dataset(test_data, field_name=Const.INPUT,new_field_name=Const.INPUT)

    # print(test_data[0])
    print(len(vocab))
    return vocab, train_data, dev_data, test_data

if __name__ == "__main__":
    vocab, train_data, dev_data, test_data = get_data()

    with open("data.bin", "wb") as f:
        pickle.dump((vocab, train_data, dev_data, test_data), f)