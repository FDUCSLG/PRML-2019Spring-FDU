import pickle
import os
import string
import re
from sklearn.datasets import fetch_20newsgroups
from fastNLP import DataSet
from fastNLP import Vocabulary

from config import Config


def fetch_dataset(data_path):
    train_data = fetch_20newsgroups(subset='train', data_home=data_path)
    test_data = fetch_20newsgroups(subset='test', data_home=data_path)
    print('Samples:', len(train_data.data), len(test_data.data))
    print('Categories:', len(train_data.target_names), len(test_data.target_names))

    return train_data, test_data


def get_dataset(raw_data):
    data_dict = {
        "input": raw_data.data,
        "target": raw_data.target
    }
    dataset = DataSet(data=data_dict)

    # ignore string.punctuation
    dataset.apply(lambda x: x['input'].translate(str.maketrans("", "", string.punctuation)), new_field_name='input')
    # string.whitespace -> space
    dataset.apply(lambda x: re.sub('[' + string.whitespace + ']', ' ', x['input']), new_field_name='input')
    # lower case & split by space
    dataset.apply(lambda x: x['input'].lower().split(' '), new_field_name='input')

    # target: int
    dataset.set_input('input')
    dataset.set_target('target')
    return dataset


def get_vocabulary(dataset):
    vocabulary = Vocabulary(min_freq=2, unknown='<oov>', padding='<pad>')
    # vocabulary.add_word('<eos>')
    # vocabulary.add_word('<start>')

    dataset.apply(lambda x: [vocabulary.add(word) for word in x['input']])
    vocabulary.build_vocab()

    print('pad:', vocabulary.to_index('<pad>'))
    print('Vocab size:', len(vocabulary))
    return vocabulary


def get_data(train_raw, test_raw):
    train_data, dev_data = get_dataset(train_raw).split(0.2)
    test_data = get_dataset(test_raw)

    vocabulary = get_vocabulary(train_data)
    train_data.apply(lambda x: [vocabulary.to_index(word) for word in x['input']], new_field_name='input')
    dev_data.apply(lambda x: [vocabulary.to_index(word) for word in x['input']], new_field_name='input')
    test_data.apply(lambda x: [vocabulary.to_index(word) for word in x['input']], new_field_name='input')

    print("Sizes:", len(train_data), len(dev_data), len(test_data))
    print("Sample:", train_data[0])

    return train_data, dev_data, test_data, vocabulary


def dump_dataset(config):
    train_raw, test_raw = fetch_dataset(config.data_path)
    train_data, dev_data, test_data, vocabulary = get_data(train_raw, test_raw)

    # dump data
    pickle.dump(train_data, open(os.path.join(config.data_path, config.train_name), "wb"))
    pickle.dump(dev_data, open(os.path.join(config.data_path, config.dev_name), "wb"))
    pickle.dump(test_data, open(os.path.join(config.data_path, config.test_name), "wb"))
    pickle.dump(vocabulary, open(os.path.join(config.data_path, config.vocabulary_name), "wb"))


if __name__ == "__main__":
    config = Config()
    dump_dataset(config)
