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

def document2line(doc):
    new_doc=""
    for i in range(len(doc)):
        if(doc[i]=='\t' or doc[i]=='\n'):
            new_doc=new_doc+" "
        else:
            new_doc=new_doc+doc[i]
    lst=new_doc.split(" ")
    new_doc=""
    for ele in lst:
        if ele!="":
            new_doc=new_doc+" "+ele.strip()  ## 此处的每一个单词也要将空格.strip()掉
    new_doc=new_doc.strip()
    return new_doc
#print(document2line("hello\nhello\n\nnihao   

def run_cnn():
    dataset_train_p2,dataset_test_p2=get_text_classification_datasets()

    line_len=len(dataset_train_p2.data)
    with open("formalized_train_data.csv","w") as file:
        for i in range(line_len):
            file.write(document2line(dataset_train_p2.data[i])+"\t"+str(dataset_train_p2.target[i])+'\n')
        file.close()

    line_len=len(dataset_test_p2.data)
    with open("formalized_test_data.csv","w") as file2:
        for i in range(line_len):
            file2.write(document2line(dataset_test_p2.data[i])+"\t"+str(dataset_test_p2.target[i])+'\n')
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

    #train_dataset[0],test_dataset[0]

    from fastNLP import Vocabulary

    # 使用Vocabulary类统计单词，并将单词序列转化为数字序列
    vocab = Vocabulary(min_freq=2).from_dataset(train_dataset, field_name='words')
    vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
    vocab.index_dataset(test_dataset,field_name='words',new_field_name='words')
    #train_dataset[0],test_dataset[0]

    # 将label转为整数，并设置为 target
    train_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    test_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

    #train_dataset[0],test_dataset[0]

    from fastNLP.models import CNNText
    embed_dim=50
    model = CNNText((len(vocab),embed_dim), num_classes=4, padding=2, dropout=0.1)
    model

    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

    # 定义trainer并进行训练
    trainer = Trainer(model=model, train_data=train_dataset, dev_data=test_dataset,
                  loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()



# 定义 Recurrent Network 模型
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



def run_rnn():
    dataset_train_p2,dataset_test_p2=get_text_classification_datasets()
    line_len=len(dataset_train_p2.data)
    with open("formalized_train_data.csv","w") as file:
        for i in range(line_len):
            file.write(document2line(dataset_train_p2.data[i])+"\t"+str(dataset_train_p2.target[i])+'\n')
        file.close()

    line_len=len(dataset_test_p2.data)
    with open("formalized_test_data.csv","w") as file2:
        for i in range(line_len):
            file2.write(document2line(dataset_test_p2.data[i])+"\t"+str(dataset_test_p2.target[i])+'\n')
        file2.close()

    loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
    train_dataset = loader.load("./formalized_train_data.csv")
    test_dataset = loader.load("./formalized_test_data.csv")

    train_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    train_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

    test_dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    test_dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

    from fastNLP import Vocabulary

    # 使用Vocabulary类统计单词，并将单词序列转化为数字序列
    vocab = Vocabulary(min_freq=2).from_dataset(train_dataset, field_name='words')
    vocab.index_dataset(train_dataset, field_name='words',new_field_name='words')
    vocab.index_dataset(test_dataset,field_name='words',new_field_name='words')
    # 将label转为整数，并设置为 target
    train_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    test_dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

    model = Rnn(len(vocab), 128, 128, 2, 4)  # 图片大小是28x28
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        model = model.cuda()

    trainer = Trainer(model=model, train_data=train_dataset, dev_data=test_dataset,
                  loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()


print("press 1 to run cnn and press 2 to run rnn:")
while(True):
    if input()=="1":
        run_cnn()
    else:
        run_rnn()
