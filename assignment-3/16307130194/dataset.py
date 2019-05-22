from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import numpy as np


def get_dataset(data_path, dataset='large'):
    poetry = []
    if dataset == 'small':
        with open(data_path, 'r', encoding='utf-8') as f:
            poem = ''
            for line in f:
                if len(line) <= 1:
                    ins = Instance(text=poem)
                    poetry.append(ins)
                    poem = ''
                else:
                    poem += line.strip('\n')
    else:
        data = np.load(data_path)
        data, ix2word = data['data'], data['ix2word'].item()
        for d in data
        poetry = data['data']

    print(poetry[0])

    data = DataSet(data=poetry)
    data.apply(lambda x: len(x['text']), new_field_name='length')
    # print(data[0])

    train_data, dev_data = data.split(dev_ratio=0.2)
    print(train_data.get_length(), dev_data.get_length())
    # print(train_data[0])
    # print(dev_data[0])

    vocabulary = Vocabulary(min_freq=2, unknown='<oov>', padding='<pad>')
    vocabulary.add_word('<EOS>')
    train_data.apply(lambda x: [vocabulary.add(char) for char in x['text']])
    vocabulary.build_vocab()

    train_data.apply(lambda x: [vocabulary.to_index(char) for char in x['text']], new_field_name='chars')
    dev_data.apply(lambda x: [vocabulary.to_index(char) for char in x['text']], new_field_name='chars')
    print(dev_data[0])

    return train_data, dev_data, vocabulary
