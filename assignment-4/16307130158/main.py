import sys
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import sklearn
import string
import torch
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP import Const
from fastNLP import AccuracyMetric
from fastNLP import Tester
from CNN import CNN_model
from RNN import RNN_model
from RNN import RNN_single_gate
from train import cnn_train, rnn_train

raw_train = fetch_20newsgroups(subset = 'train')
raw_test = fetch_20newsgroups(subset = 'test')
print('[info] Data already loaded.')
def preprocess():
    train_set = DataSet()
    for i in range(len(raw_train.data)):
        train_set.append(Instance(sentence=raw_train.data[i], target=int(raw_train.target[i])))

    train_set.apply(lambda x: x['sentence'].translate(str.maketrans("", "", string.punctuation)).lower(), new_field_name='sentence')
    train_set.apply(lambda x: x['sentence'].split(), new_field_name='words')
    train_set.apply(lambda x: len(x['words']), new_field_name='seq_len')

    test_set = DataSet()
    for i in range(len(raw_test.data)):
        test_set.append(Instance(sentence=raw_test.data[i], target=int(raw_test.target[i])))

    test_set.apply(lambda x: x['sentence'].translate(str.maketrans("", "", string.punctuation)).lower(), new_field_name='sentence')
    test_set.apply(lambda x: x['sentence'].split(), new_field_name='words')
    test_set.apply(lambda x: len(x['words']), new_field_name='seq_len')

    vocab = Vocabulary(min_freq=10)
    train_set.apply(lambda x: [vocab.add(word) for word in x['words']])
    test_set.apply(lambda x: [vocab.add(word) for word in x['words']])
    vocab.build_vocab()
    vocab.index_dataset(train_set, field_name='words', new_field_name='words')
    vocab.index_dataset(test_set, field_name='words', new_field_name='words')

    return train_set, test_set, vocab


train_set, test_set, vocab = preprocess()
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


if __name__ == '__main__':
    metrics = AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)

    # CNN
    device = torch.device("cuda")
    model_cnn = CNN_model(vocab_size=len(vocab), embedding_dim=150, num_classes=20, padding=2, dropout=0.1)
    # model_cnn.load_state_dict(torch.load('./rec_cnn_dim_50/cnn_state.pth'))
    model_cnn.to(device)
    cnn_train(epoch=20, data=train_set, model=model_cnn)
    # tester = Tester(data=test_set, model=model_cnn, metrics=AccuracyMetric())
    # tester.test()




    # # RNN
    # device = torch.device("cuda")
    # model_rnn = RNN_model(vocab_size=len(vocab), embedding_dim=130, hidden_dim=130, num_classes=20)
    # # model_rnn.load_state_dict(torch.load('./rnn_state.pth'))
    # model_rnn.to(device)
    # rnn_train(epoch=20, data=train_set, model = model_rnn)
    # # tester = Tester(data=test_set, model=model_rnn, metrics=AccuracyMetric())
    # # tester.test()



