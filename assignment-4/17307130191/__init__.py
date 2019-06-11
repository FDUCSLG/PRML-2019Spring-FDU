import os
os.sys.path.append('../../assignment-2')
from handout import get_text_classification_datasets

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import BucketSampler
from fastNLP import Batch
import re
import string

vocabsize = 0
maxlen = 60


traindata, testdata = get_text_classification_datasets()

def pre(x):
    data = re.sub(r'[^a-zA-Z0-9\s]','',x)
    data = re.sub(r'['+string.whitespace+']+',' ',data)
    data = re.split(r' +', data.strip().lower())
    if len(data) >= maxlen:
        return data[:maxlen]
    else:
        return data

def preprocess(batch=16):
    raw_data1 = []
    raw_data2 = []

    for i in range(len(traindata.data)):
        raw_data1.append(Instance(sentence=traindata.data[i], label=int(traindata.target[i])))
    trainset = DataSet(raw_data1)
    trainset.apply(lambda x: pre(x['sentence']), new_field_name='words')

    for i in range(len(testdata.data)):
        raw_data2.append(Instance(sentence=testdata.data[i], label=int(testdata.target[i])))
    testset = DataSet(raw_data2)
    testset.apply(lambda x: pre(x['sentence']), new_field_name='words')

    global vocab
    vocab = Vocabulary(min_freq=1).from_dataset(trainset, testset, field_name='words')
    vocab.index_dataset(trainset, testset, field_name='words', new_field_name='words')
    trainset.set_input('words')
    testset.set_input('words')

    trainset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    testset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

    trainset.apply(lambda x: len(x['words']), new_field_name='seq_len')
    testset.apply(lambda x: len(x['words']), new_field_name='seq_len')

    global  vocabsize
    vocabsize = len(vocab)
    sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
    train_batch = Batch(batch_size=batch, dataset=trainset, sampler=sampler)
    test_batch = Batch(batch_size=batch, dataset=testset, sampler=sampler)

    return train_batch, test_batch, vocabsize