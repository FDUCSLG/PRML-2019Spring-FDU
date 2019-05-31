from fastNLP import Trainer
from fastNLP import Adam
from fastNLP import EarlyStopCallback
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
from fastNLP import Tester

import os
import pickle

from config import Config
from model import CNN, CNN_w2v, RNN, LSTM, LSTM_maxpool, RCNN
from utils import TimingCallback


def train(config):
    train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    vocabulary = pickle.load(open(os.path.join(config.data_path, config.vocabulary_name), "rb"))
    # load w2v data
    weight = pickle.load(open(os.path.join(config.data_path, config.weight_name), "rb"))

    if config.task_name == "lstm":
        text_model = LSTM(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         output_dim=config.class_num, hidden_dim=config.hidden_dim,
                         num_layers=config.num_layers, dropout=config.dropout)
    elif config.task_name == "lstm_maxpool":
        text_model = LSTM_maxpool(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         		  output_dim=config.class_num, hidden_dim=config.hidden_dim,
                         		  num_layers=config.num_layers, dropout=config.dropout)
    elif config.task_name == "rnn":
        text_model = RNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         output_dim=config.class_num, hidden_dim=config.hidden_dim,
                         num_layers=config.num_layers, dropout=config.dropout)
    elif config.task_name == "cnn":
        text_model = CNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         class_num=config.class_num, kernel_num=config.kernel_num,
                         kernel_sizes=config.kernel_sizes, dropout=config.dropout,
                         static=config.static, in_channels=config.in_channels)
    elif config.task_name == "cnn_w2v":
        text_model = CNN_w2v(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                             class_num=config.class_num, kernel_num=config.kernel_num,
                             kernel_sizes=config.kernel_sizes, dropout=config.dropout,
                             static=config.static, in_channels=config.in_channels,
                             weight=weight)
    elif config.task_name == "rcnn":
        text_model = RCNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim, 
                          output_dim=config.class_num, hidden_dim=config.hidden_dim, 
                          num_layers=config.num_layers, dropout=config.dropout)
    optimizer = Adam(lr=config.lr, weight_decay=config.weight_decay)
    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)
    accuracy = AccuracyMetric(pred='output', target='target')

    trainer = Trainer(train_data=train_data, model=text_model, loss=CrossEntropyLoss(),
                      batch_size=config.batch_size, check_code_level=0,
                      metrics=accuracy, n_epochs=config.epoch,
                      dev_data=dev_data, save_path=config.save_path,
                      print_every=config.print_every, validate_every=config.validate_every,
                      optimizer=optimizer, use_tqdm=False,
                      device=config.device, callbacks=[timing, early_stop])
    trainer.train()

    # test result
    tester = Tester(test_data, text_model, metrics=accuracy)
    tester.test()


if __name__ == "__main__":
    config = Config()
    # print configs
    with open('config.py', 'r') as f:
        for line in f:
            print(line, end='')
    train(config)

