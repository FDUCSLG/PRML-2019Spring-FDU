import sys,string,re
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from sklearn.datasets import fetch_20newsgroups
import config
import pickle

def construct_dataset(dataset):
    dataset_ = DataSet()
    for sentence,target in zip(dataset.data, dataset.target):
        instance = Instance()
        instance['raw_sentence'] = sentence
        instance['target'] = int(target)
        dataset_.append(instance)

    dataset_.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x['raw_sentence']), new_field_name = 'sentence') #忽略标点
    dataset_.apply(lambda x: re.sub('[%s]' % re.escape(string.whitespace), ' ', x['sentence']), new_field_name = 'sentence') #将空格、换行符等空白替换为空格
    dataset_.apply(lambda x: x['sentence'].lower(), new_field_name = 'sentence') #转换为小写
    dataset_.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='input')
    return dataset_

def Get_Data_Vocab():
    dataset_train = fetch_20newsgroups(subset='train', data_home='../../..')
    dataset_test = fetch_20newsgroups(subset='test', data_home='../../..')

    train_data_raw = construct_dataset(dataset_train)
    test_data = construct_dataset(dataset_test)
    vocab = Vocabulary(min_freq=10).from_dataset(train_data_raw, field_name='input')
    vocab.index_dataset(train_data_raw, field_name='input',new_field_name='input')
    vocab.index_dataset(test_data, field_name='input',new_field_name='input')
    train_data_raw.set_input("input")
    train_data_raw.set_target("target")
    test_data.set_input("input")
    test_data.set_target("target")
    dev_data, train_data = train_data_raw.split(0.8)
    
    return vocab, train_data, dev_data, test_data

if __name__ == "__main__":
    vocab, train_data, dev_data, test_data = Get_Data_Vocab()
    pickle.dump(vocab, open(config.vocab_path, "wb"))  
    pickle.dump(train_data, open(config.train_data_path, "wb"))
    pickle.dump(dev_data, open(config.dev_data_path, "wb"))
    pickle.dump(test_data, open(config.test_data_path, "wb"))