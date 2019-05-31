import pickle
import numpy as np
import torch

from fastNLP.core import Adam
from model import MyCNNText, MyLSTMText, MyBLSTMText
from fastNLP import CrossEntropyLoss, AccuracyMetric, Trainer, Tester
from fastNLP.core import EarlyStopCallback
from data import read_data

def cnn_text():
    w = pickle.load(open("weight.bin", "rb"))

    (vocab, train_data, dev_data, test_data) = read_data()

    model_cnn = MyCNNText(class_num=4, vocab_size=len(vocab), embed_weights=w)
    loss = CrossEntropyLoss()
    metrics = AccuracyMetric()
    trainer = Trainer(model=model_cnn,
                    train_data=train_data,
                    dev_data=dev_data,
                    batch_size=32,
                    print_every=10,
                    use_tqdm=False,
                    device='cuda:0',
                    save_path="./cnn_model",
                    loss=loss,
                    metrics=metrics)

    trainer.train()

    tester = Tester(test_data, model_cnn, metrics=AccuracyMetric())
    tester.test()

def lstm_text():
    w = pickle.load(open("weight.bin", "rb"))

    (vocab, train_data, dev_data, test_data) = read_data()

    model_lstm = MyLSTMText(class_num=4, vocab_size=len(vocab), dropout=0.5, embed_weights=w)
    loss = CrossEntropyLoss()
    metrics = AccuracyMetric()
    trainer = Trainer(model=model_lstm,
                    train_data=train_data,
                    dev_data=dev_data,
                    print_every=10,
                    use_tqdm=False,
                    device='cpu',
                    save_path="./lstm_model",
                    loss=loss,
                    metrics=metrics)
                    # callbacks=[EarlyStopCallback(10)])

    trainer.train()

    tester = Tester(test_data, model_lstm, metrics=AccuracyMetric())
    tester.test()

def bilstm_text():
    w = pickle.load(open("weight.bin", "rb"))

    (vocab, train_data, dev_data, test_data) = read_data()

    model_lstm = MyBLSTMText(class_num=4, vocab_size=len(vocab), dropout=0.5, embed_weights=w)
    loss = CrossEntropyLoss()
    metrics = AccuracyMetric()
    trainer = Trainer(model=model_lstm,
                    train_data=train_data,
                    dev_data=dev_data,
                    optimizer=Adam(lr=0.0015),
                    print_every=10,
                    use_tqdm=False,
                    device='cuda:0',
                    save_path="./lstm_model",
                    loss=loss,
                    metrics=metrics)
                    # callbacks=[EarlyStopCallback(10)])

    trainer.train()

    tester = Tester(test_data, model_lstm, metrics=AccuracyMetric())
    tester.test()

if __name__ == "__main__":
    # cnn_text()
    # lstm_text()
    bilstm_text()
