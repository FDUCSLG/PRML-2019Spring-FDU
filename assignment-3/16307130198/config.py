import numpy as np
import os


class Config(object):
    train_data_path = "./dataset/train_data.pkl"
    validate_data_path = "./dataset/validate_data.pkl"
    vocab = "./dataset/vocab.pkl"
    use_gpu = False
    epoch = 100
    batch_size = 128
    maxlen = 128
    max_gen_len = 200
    validate_every = 300
    patience = 10
    embedding_dim = 512
    hidden_dim = 512
    optimizer = "Adam"
    learning_rate = 1e-3
    model_name = "PoetryModel"
    save_model_path = "./model_log"
