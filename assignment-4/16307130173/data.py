import sys
from handout import get_text_classification_datasets
import sklearn

import torch
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP.core import Const

import string

def get_dataset():
    raw_train, raw_test = get_text_classification_datasets()

    def transfer(st):
        st = st.lower()
        for i in string.whitespace:
            st.replace(i, '')
        for i in string.punctuation:
            st.replace(i, ' ')

        return st
    
    def preprocess():
        train_set = DataSet()
        for i in range(len(raw_train['data'])):
            di = transfer(raw_train['data'][i])
            train_set.append(Instance(sentence = di, target=int(raw_train['target'][i])))
            
        train_set.apply(lambda x: x['sentence'].lower(), new_field_name='sentence')
        train_set.apply(lambda x: x['sentence'].split(), new_field_name='words')
        train_set.apply(lambda x: len(x['words']), new_field_name='seq_len')

        test_set = DataSet()
        for i in range(len(raw_test['data'])):
            di = transfer(raw_test['data'][i])
            test_set.append(Instance(sentence = di, target=int(raw_test['target'][i])))
            
        test_set.apply(lambda x: x['sentence'].lower(), new_field_name='sentence')
        test_set.apply(lambda x: x['sentence'].split(), new_field_name='words')
        test_set.apply(lambda x: len(x['words']), new_field_name='seq_len')

        word_dict = Vocabulary(min_freq=2)
        train_set.apply(lambda x: [word_dict.add(word) for word in x['words']])
        test_set.apply(lambda x: [word_dict.add(word) for word in x['words']])
        word_dict.build_vocab()
        word_dict.index_dataset(train_set, field_name='words', new_field_name='words')
        word_dict.index_dataset(test_set, field_name='words', new_field_name='words')

        return train_set, test_set, word_dict



    train_set, test_set, word_dict = preprocess()
    train_set.rename_field('words', Const.INPUT)
    train_set.rename_field('seq_len', Const.INPUT_LEN)
    train_set.rename_field('target', Const.TARGET)
    test_set.rename_field('words', Const.INPUT)
    test_set.rename_field('seq_len', Const.INPUT_LEN)
    test_set.rename_field('target', Const.TARGET)


    
    train_set.set_input(Const.INPUT, Const.INPUT_LEN)
    train_set.set_target(Const.TARGET)
    test_set.set_input(Const.INPUT, Const.INPUT_LEN)
    test_set.set_target(Const.TARGET)

    return train_set, test_set, word_dict
