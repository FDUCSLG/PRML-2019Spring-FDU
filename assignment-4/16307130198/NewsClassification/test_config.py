import numpy as np
import os


class Config(object):
    test_data_path = "./dataset/test_data.pkl"
    vocab = "./dataset/vocab.pkl"
    batch_size = 8
    embedding_dim = 256
    #model_name = "MyBertModel"
    model_name = "B_LSTMModel"
    #model_name = "LSTMModel"
    #model_name = "CNNModel"
    model_path = "./model_log/best_DataParallel_acc_2019-06-01-00-00-55" 
    class_num = 20

