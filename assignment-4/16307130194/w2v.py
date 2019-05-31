import pickle
import os
import string
import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from gensim.models.word2vec import Word2Vec 
from fastNLP import DataSet
from fastNLP import Vocabulary

from config import Config


def dump_w2v(config):
    train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    vocabulary = pickle.load(open(os.path.join(config.data_path, config.vocabulary_name), "rb"))

    print(train_data[0])

    train_data.apply(lambda x: [vocabulary.to_word(idx) for idx in x['input']], new_field_name='input')
    dev_data.apply(lambda x: [vocabulary.to_word(idx) for idx in x['input']], new_field_name='input')
    test_data.apply(lambda x: [vocabulary.to_word(idx) for idx in x['input']], new_field_name='input')

    print(train_data[0])
    dataset = []
    train_data.apply(lambda x: dataset.append(x['input']))

    print(dataset[0])

    model= Word2Vec(window=1, min_count=1, size=config.embed_dim)
    model.build_vocab(dataset)
    model.train(dataset, total_examples=model.corpus_count, epochs=model.iter)

    weight = np.zeros([len(vocabulary), config.embed_dim])
    for i in range(1, len(vocabulary)):
        weight[i] = model[vocabulary.to_word(i)]

    pickle.dump(weight, open(os.path.join(config.data_path, config.weight_name), "wb"))


if __name__ == "__main__":
    config = Config()
    dump_w2v(config)
