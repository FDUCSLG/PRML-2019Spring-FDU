import os
import torch
from fastNLP import Tester
from fastNLP.core.metrics import AccuracyMetric
from model import CNN,RNN
import pickle
import config

def load_model(model, save_path, model_name):
    model_path = os.path.join(save_path, model_name)
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)

if __name__ == "__main__":

    vocab = pickle.load(open(config.vocab_path, 'rb'))
    test_data = pickle.load(open(config.test_data_path, 'rb'))
    if config.model == "CNN":
        model = CNN(len(vocab), config.intput_size, config.class_num)
    elif config.model == "RNN":
        model = RNN(len(vocab), config.intput_size, config.hidden_size, config.class_num,config.rnn_type)

    load_model(model, 
              save_path=config.save_path, 
              model_name="RNNmax")
    metrics = AccuracyMetric(pred="output", target="target")
    tester = Tester(test_data, model, metrics=metrics, device='cuda:0')
    tester.test()