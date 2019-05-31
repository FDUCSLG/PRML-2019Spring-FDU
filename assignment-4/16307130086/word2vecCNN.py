import os
import fastNLP
import gensim
os.sys.path.append('..')
# from handout import get_text_classification_datasets
# trainData,testData = get_text_classification_datasets()
from fastNLP import Instance
from fastNLP import DataSet
from fastNLP import Vocabulary
# from fastNLP.models import CNNText
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from fastNLP.core.const import Const as C
from fastNLP.modules import encoder
from torch.autograd import Variable

trainSet = fetch_20newsgroups(subset='train')
testSet = fetch_20newsgroups(subset='test')

testData = {
    "data": testSet['data'],
    "target": testSet['target']
}

trainData = {
    "data": trainSet['data'],
    "target": trainSet['target']
}
trainData = DataSet(trainData)
testData = DataSet(testData)

#     allData[i].append("<pad>")

trainData.apply(lambda x: x['data'].lower(), new_field_name='sentence')
trainData.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)
vocab = Vocabulary(min_freq=2)
vocab = vocab.from_dataset(trainData, field_name='words')
allData = trainData['words']
for i in range(len(allData)):
    allData[i].append("<pad>")
    allData[i].append("<unk>")
W2Vmodel = gensim.models.Word2Vec(allData, min_count=2, iter=15, size=300)
W2Vmodel.wv.save_word2vec_format("w2v.model")
model = gensim.models.KeyedVectors.load_word2vec_format('w2v.model')
weights = torch.FloatTensor(model.vectors)
#change to index

vocab.index_dataset(trainData, field_name='words',new_field_name='words')
trainData.set_target('target')
train_data, dev_data = trainData.split(0.2)

from torch.nn import Parameter
class w2vCNNText(torch.nn.Module):
    
    def __init__(self, init_embed,
                 num_classes,
                 vocab,
                 pretrained_weight,
                 kernel_nums=(10,10,10,10),
                 kernel_sizes=(2, 3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(w2vCNNText, self).__init__()
        
        # no support for pre-trained embedding currently
        self.word_embeds = nn.Embedding(init_embed[0],init_embed[1])
        pretrained_weight = np.array(pretrained_weight)
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=init_embed[1],
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
    def forward(self, words, seq_len=None):
        batch_size = words.size()[0]
        seq_len = words.size()[1]
#         w2vWords = torch.DoubleTensor(batch_size, seq_len, self.embedding_dim)
#         for i in range(batch_size):
#             for j in range(seq_len):
#                 word = self.vocab.to_word(words[i][j].item())
#                 if word in self.W2Vmodel:
#                     w2vWords[i][j] = torch.tensor(self.W2Vmodel[word])
#                 else:
# #                     print(word)
#                     w2vWords[i][j] = torch.zeros(self.embedding_dim)
#         w2vWords = w2vWords.float()
        x = self.word_embeds(words)
        x = self.conv_pool(x)  
        x = self.dropout(x)
        x = self.fc(x)  
        return {C.OUTPUT: x}
    
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
    
trainData.apply(lambda x: x['data'].lower(), new_field_name='sentence')
trainData.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)
vocab = Vocabulary(min_freq=2)
vocab = vocab.from_dataset(trainData, field_name='words')
#change to index
vocab.index_dataset(trainData, field_name='words',new_field_name='words')
trainData.set_target('target')
model = w2vCNNText((len(vocab),300), num_classes=20, vocab=vocab, pretrained_weight=weights, padding=2, dropout=0.1)
train_data, dev_data = trainData.split(0.2)
trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=CrossEntropyLoss(), metrics=AccuracyMetric(), batch_size=16)
trainer.train()