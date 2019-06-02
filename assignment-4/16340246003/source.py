import os
os.sys.path.append('..')
import string
from handout import get_linear_seperatable_2d_2c_dataset,get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import math
from fastNLP.io import CSVLoader
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from fastNLP.core.const import Const as C
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
from fastNLP import Vocabulary
from fastNLP.models import CNNText




def doc_to_line(d):
    new_d=""
    for i in range(len(d)):
        if(d[i]=='\t' or d[i]=='\n'):
            new_d=new_d+" "
        else:
            new_d=new_d+d[i]
    lst=new_d.split(" ")
    new_d=""
    for e in lst:
        if e!="":
            new_d=new_d+" "+e.strip()
    new_d=new_d.strip()
    return new_d





def run_cnn():
    dataset_train_p2,dataset_test_p2=get_text_classification_datasets()
    
    line_len=len(dataset_train_p2.data)
    with open("formalized_train_data.csv","w") as file:
        for i in range(line_len):
            file.write(doc_to_line(dataset_train_p2.data[i])+"\t"+str(dataset_train_p2.target[i])+'\n')
        file.close()
    
    line_len=len(dataset_test_p2.data)
    with open("formalized_test_data.csv","w") as file2:
        for i in range(line_len):
            file2.write(doc_to_line(dataset_test_p2.data[i])+"\t"+str(dataset_test_p2.target[i])+'\n')
        file2.close()
    
    loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
    train_dataset = loader.load("./formalized_train_data.csv")
    test_dataset = loader.load("./formalized_test_data.csv")

    os.remove("./formalized_train_data.csv")
    os.remove("./formalized_test_data.csv")

    train_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    train_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

    test_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    test_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

    vocab = Vocabulary(min_freq=2).from_dataset(train_dataset, field_name='words')
    vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
    vocab.index_dataset(test_dataset,field_name='words',new_field_name='words')
    train_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    test_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    embed_dim=2048 #50
    model = CNNText((len(vocab),embed_dim), num_classes=4, padding=2, dropout=0.1)
    model
    trainer = Trainer(model=model, train_data=train_dataset, dev_data=test_dataset,
                      loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()





def run_rnn():
    dataset_train_p2,dataset_test_p2=get_text_classification_datasets()
    line_len=len(dataset_train_p2.data)
    with open("formalized_train_data.csv","w") as file:
        for i in range(line_len):
            file.write(doc_to_line(dataset_train_p2.data[i])+"\t"+str(dataset_train_p2.target[i])+'\n')
        file.close()
    
    line_len=len(dataset_test_p2.data)
    with open("formalized_test_data.csv","w") as file2:
        for i in range(line_len):
            file2.write(doc_to_line(dataset_test_p2.data[i])+"\t"+str(dataset_test_p2.target[i])+'\n')
        file2.close()
    
    loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
    train_dataset = loader.load("./formalized_train_data.csv")
    test_dataset = loader.load("./formalized_test_data.csv")

    train_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    train_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

    test_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    test_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)
    vocab = Vocabulary(min_freq=2).from_dataset(train_dataset, field_name='words')
    vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
    vocab.index_dataset(test_dataset,field_name='words',new_field_name='words')
    train_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    test_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    embed_dim=1024
    hidden_dim=128
    layer=4
    
    model = Rnn(len(vocab), embed_dim, hidden_dim, layer,4 )
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    
    trainer = Trainer(model=model, train_data=train_dataset, dev_data=test_dataset,
                      loss=CrossEntropyLoss(), n_epochs=100, metrics=AccuracyMetric())
    trainer.train()

class Rnn(nn.Module):
    def __init__(self, vocab_size, embd_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.embd_dim = embd_dim
        self.word_embd = nn.Embedding(vocab_size, embd_dim)
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.lstm = nn.LSTM(embd_dim, hidden_dim, n_layer, batch_first=True, )
        self.classifier = nn.Linear(hidden_dim, n_class)
    
    def forward(self, words):
        x = self.word_embd(words)
        x, _ = self.lstm(x)
        out = self.classifier(x[:, -1, :])
        return {C.OUTPUT: out}
    
    def predict(self, words):
        output = self(words)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}


run_cnn()
#run_rnn()
