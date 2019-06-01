import numpy as np
import os


class Config(object):
    train_data_path = "./dataset/train_data.pkl"
    validate_data_path = "./dataset/validate_data.pkl"
    test_data_path = "./dataset/test_data.pkl"
    embedding_weight_path = "./dataset/embedding_weight.pkl"
    vocab = "./dataset/vocab.pkl"
    epoch = 100
    batch_size = 32
    validate_every = 50
    patience = 10
    embedding_dim = 256
    optimizer = "Adam"
    learning_rate = 1e-3
    #model_name = "MyBertModel"
    model_name = "B_LSTMModel"
    #model_name = "LSTMModel"
    #model_name = "CNNModel"
    save_model_path = "./model_log"
    use_word2vec = True
    class_num = 20
