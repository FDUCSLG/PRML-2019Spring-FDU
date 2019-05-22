from data import *
from model import *


def TrainModel():
    vocab, train_data = pre_process()
    vocab_size = len(vocab)
    LSTM = Poetry(vocab_size, embedding_dim=128, hidden_dim=128)
    LSTM.train(train_data[:][:-1], train_data[:][1:], lr=0.005, epoch=3)


if __name__ == '__main__':
    TrainModel()
