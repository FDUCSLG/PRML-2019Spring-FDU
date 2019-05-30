import sys
sys.path.append("..")

from sklearn.datasets import fetch_20newsgroups
from data.dataset import *
from fastNLP import Vocabulary
from fastNLP import Instance
from fastNLP import DataSet
import pandas as pd
from fastNLP import Callback

# build_env()
from fastNLP.io import CSVLoader


# fastNLP version
def data_preprocess():
    dataset_train,dataset_test=get_text_classification_datasets()
    cont=dataset_train.data
    targ=dataset_train.target
    
    dataset = DataSet()
    num=len(targ)
    
    for i in range(num):
        instance=Instance(raw_sentence=cont[i],label=int(targ[i]))
        dataset.append(instance)

    print('finish load')

    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')

    # 使用Vocabulary类统计单词，并将单词序列转化为数字序列
    vocab = Vocabulary(min_freq=15).from_dataset(dataset, field_name='words')
    vocab.index_dataset(dataset, field_name='words',new_field_name='words')

    # 将label转为整数
    dataset.apply(lambda x: int(x['label']), new_field_name='target')

    dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    dropped_dataset = dataset.drop(lambda ins:ins['seq_len']>500, inplace=False)
    dropped_dataset.set_input('words')
    # dropped_dataset.set_target('label')
    dropped_dataset.set_target('target')

    train_data, test_data = dropped_dataset.split(0.1)
    train_data, dev_data = train_data.split(0.1)

    return train_data,dev_data,test_data,len(vocab),vocab
    
def test_data_(vocab):
    dataset_train,dataset_test=get_text_classification_datasets()
    cont=dataset_test.data
    targ=dataset_test.target
    
    dataset = DataSet()
    num=len(targ)
    
    for i in range(num):
        instance=Instance(raw_sentence=cont[i],label=int(targ[i]))
        dataset.append(instance)

    print('finish load')

    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')

    # 使用Vocabulary类统计单词，并将单词序列转化为数字序列
    vocab.index_dataset(dataset, field_name='words',new_field_name='words')

    # 将label转为整数
    dataset.apply(lambda x: int(x['label']), new_field_name='target')

    dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    dropped_dataset = dataset.drop(lambda ins:ins['seq_len']>500, inplace=False)
    dropped_dataset.set_input('words')
    dropped_dataset.set_target('label')
    
    return dropped_dataset
# data_preprocess()
