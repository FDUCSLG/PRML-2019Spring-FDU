import fastNLP
from fastNLP import Instance,Vocabulary,Const,AccuracyMetric,Tester,Trainer
import fastNLP.core.losses
import math
import random
import numpy as np
import torch.nn as nn
import torch
import argparse
import utils
import models
from sklearn.datasets import fetch_20newsgroups
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="RNN", dest="model", help="Name for model")
parser.add_argument("--name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="name", help="Name for this task")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")  
parser.add_argument("--embed_dim", default=128, dest="embed_dim", type=int, help="embed_dim")
parser.add_argument("--layers", default=1, dest="layers", type=int, help="layers")
parser.add_argument("--hidden_size", default=128, dest="hidden_size", type=int, help="hidden_size")
parser.add_argument("--dropout", default=0.5, dest="dropout", type=float,
                    help="dropout")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--batch-size", default=32, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--num_epochs", default=20, dest="num_epochs", type=int,
                    help="Number of full passes through training set")   
parser.add_argument("--lr", default=1e-3, dest="lr", type=float,help="learning rate")
parser.add_argument("--gpu", default=True, dest="gpu", action="store_false", help="gpu")
options = parser.parse_args()
print(options)

categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
#raw_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
#raw_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
raw_train = fetch_20newsgroups(subset='train',data_home="./")
raw_test = fetch_20newsgroups(subset='test',  data_home="./")

#print(len(raw_train.data),len(raw_test.data),Const.OUTPUT)

train_set=utils.make_dataset(raw_train)
test_data=utils.make_dataset(raw_test)

vocab = Vocabulary(min_freq=10).from_dataset(train_set, field_name='words')
vocab.index_dataset(train_set, field_name='words',new_field_name='words')
vocab.index_dataset(test_data, field_name='words',new_field_name='words')
train_data, dev_data = train_set.split(0.1)
print(len(train_data), len(dev_data), len(test_data),len(vocab))

embed=models.Embedding(len(vocab),options.embed_dim)
if options.model=="RNN":
    model=models.RNNText(embed,options.hidden_size,20,dropout=options.dropout,layers=options.layers)
elif options.model=="CNN":
    model=models.CNNText(embed,20,dropout=options.dropout,padding=vocab.padding_idx)
elif options.model=="CNNKMAX":
     model=models.CNN_KMAX(embed,20,dropout=options.dropout,padding=vocab.padding_idx)
     
criterion=fastNLP.core.losses.CrossEntropyLoss()
if options.gpu:
    model=model.cuda()

best_model="{}_{}".format(options.name, options.model)   
optm=torch.optim.Adam(model.parameters(), lr = options.lr,weight_decay=1e-4)
    
trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data,loss=criterion,metrics=AccuracyMetric(),
            batch_size=options.batch_size, n_epochs=options.num_epochs,save_path=best_model,use_tqdm=False,print_every=len(train_data)//options.batch_size, optimizer=optm
            )
trainer.train()

tester = Tester(test_data, model, metrics=AccuracyMetric())
tester.test()
