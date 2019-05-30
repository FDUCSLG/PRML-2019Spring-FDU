# -*- coding: utf-8 -*-
import os
os.sys.path.append('..')
import math
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester
from fastNLP import CrossEntropyLoss
from fastNLP import NLLLoss
from fastNLP import AccuracyMetric
from fastNLP.core.callback import Callback, EarlyStopCallback
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP import Const
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

from model import *
from model_glove import *

def read_vocab(file_name):
    # 读入vocab文件
    with open(file_name) as f:
        lines = f.readlines()
    vocabs = []
    for line in lines:
        vocabs.append(line.strip())

    # 实例化Vocabulary
    vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    # 将vocabs列表加入Vocabulary
    vocab.add_word_lst(vocabs)
    # 构建词表
    vocab.build_vocab()
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='cnn', help="train model and test it",
                        choices=['cnn', 'cnn_glove', 'rnn', 'rnn_maxpool', 'rnn_avgpool'])
    parser.add_argument("--dataset", default='1', help="1: small dataset; 2: big dataset",
                        choices=['1', '2'])
    args = parser.parse_args()

    # 超参数
    embedding_dim = 256
    batch_size = 32
    # RNN
    hidden_dim = 256
    # CNN
    kernel_sizes = (3, 4, 5)
    num_channels = (120, 160, 200)
    acti_function = 'relu'

    learning_rate = 1e-3
    train_patience = 8
    cate_num = 4

    # GloVe
    embedding_file_path = "glove.6B.100d.txt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab = read_vocab("vocab.txt")
    print("vocabulary length:", len(vocab))
    train_data = DataSet().load("train_set")
    dev_data = DataSet().load("dev_set")
    test_data = DataSet().load("test_set")

    if(args.dataset == '1'):
        cate_num = 4
        num_channels = (48, 48, 48)
        embedding_dim = 128
        hidden_dim = 128
    elif(args.dataset == '2'):
        cate_num = 20
    
    if(args.method == 'cnn'):
        model = TextCNN(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                            kernel_sizes=kernel_sizes, num_channels=num_channels, 
                            num_classes=cate_num, activation=acti_function)
    elif(args.method == 'cnn_glove'):
        glove_embedding = EmbedLoader.load_with_vocab(embedding_file_path, vocab)
        embedding_dim = glove_embedding.shape[1]
        print("GloVe embedding_dim:", embedding_dim)

        model = TextCNN_glove(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                            kernel_sizes=kernel_sizes, num_channels=num_channels, 
                            num_classes=cate_num, activation=acti_function)
        model.embedding.load_state_dict({"weight":torch.from_numpy(glove_embedding)})
        model.constant_embedding.load_state_dict({"weight":torch.from_numpy(glove_embedding)})
        model.constant_embedding.weight.requires_grad = False
        model.embedding.weight.requires_grad = True

    elif(args.method == 'rnn'):
        embedding_dim = 128
        hidden_dim = 128
        model = BiRNNText(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                            output_dim=cate_num, hidden_dim=hidden_dim)
    elif(args.method == 'rnn_maxpool'):
        model = BiRNNText_pool(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                            output_dim=cate_num, hidden_dim=hidden_dim, pool_name="max")
    elif(args.method == 'rnn_avgpool'):
        model = BiRNNText_pool(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                            output_dim=cate_num, hidden_dim=hidden_dim, pool_name="avg")

    tester = Tester(test_data, model, metrics=AccuracyMetric())

    trainer = Trainer(
        train_data=train_data,
        model=model,
        loss=CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET),
        metrics=AccuracyMetric(),
        n_epochs=80,
        batch_size=batch_size,
        print_every=10,
        validate_every=-1,
        dev_data=dev_data,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        check_code_level=2,
        metric_key='acc',
        use_tqdm=True,
        callbacks=[EarlyStopCallback(train_patience)],
        device=device,
    )

    trainer.train()
    tester.test()


if(__name__ == "__main__"):
    main()