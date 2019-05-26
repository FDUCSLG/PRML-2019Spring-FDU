# -*- coding: utf-8 -*-
import math
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester
from torch import nn
from torch.autograd import Variable 

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_size):
        super(word_embedding, self).__init__()
        #embedding initial: N(0, 1)
        w_embeding_random_intial = np.random.normal(0,1,size=(vocab_length ,embedding_size))
        self.word_embedding = nn.Embedding(vocab_length,embedding_size)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


def init_lstm_state(batch_size, num_hiddens):
    return (torch.zeros((batch_size, num_hiddens), device = device), 
            torch.zeros((batch_size, num_hiddens), device = device))


# 初始化LSTM的门参数
def get_params(num_inputs, num_hiddens):
    def _one(shape):
        # torch.nn的默认参数初始化方法
        ts = torch.tensor(np.random.uniform(-math.sqrt(1/num_hiddens), math.sqrt(1/num_hiddens), size=shape), dtype=torch.float32, device = device)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _two():
        ts = torch.tensor(np.random.uniform(-math.sqrt(1/num_hiddens), math.sqrt(1/num_hiddens), size=(num_hiddens, )), dtype=torch.float32, device = device)
        return (_one((num_inputs+num_hiddens, num_hiddens)), 
                torch.nn.Parameter(ts, requires_grad=True))
    
    W_i, b_i = _two()  # 输入门参数
    W_f, b_f = _two()  # 遗忘门参数
    W_o, b_o = _two()  # 输出门参数
    W_c, b_c = _two()  # 候选记忆细胞参数

    return nn.ParameterList([W_i, b_i, W_f, b_f, W_o, b_o, W_c, b_c])


class MY_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MY_LSTM, self).__init__()
        self.hidden_size = hidden_size
        # self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.param = get_params(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, cell):
        (W_i, b_i, W_f, b_f, W_o, b_o, W_c, b_c) = self.param
        input_cuda = input.cuda()
        combined = torch.cat((input_cuda, hidden), 1)
        I = self.sigmoid(torch.matmul(combined, W_i) + b_i)
        F = self.sigmoid(torch.matmul(combined, W_f) + b_f)
        O = self.sigmoid(torch.matmul(combined, W_o) + b_o)
        cell_tilda = self.tanh(torch.matmul(combined, W_c) + b_c)
        cell = F * cell + I * cell_tilda
        hidden = O * self.tanh(cell)
        return hidden, cell


class RNN_model(nn.Module):
    def __init__(self, vocab_len, embedding_size, lstm_hidden_size):
        super(RNN_model,self).__init__()
        self.word_embedding_lookup = word_embedding(vocab_length=vocab_len, embedding_size=embedding_size).cuda()
        self.vocab_length = vocab_len
        self.word_embedding_size = embedding_size
        self.lstm_size = lstm_hidden_size

        self.rnn_lstm = MY_LSTM(embedding_size, lstm_hidden_size)

        #LSTM单元后的线性层
        w_shape = (lstm_hidden_size, vocab_len)
        self.W_hq = torch.nn.Parameter(torch.tensor(np.random.uniform(-math.sqrt(1/lstm_hidden_size), math.sqrt(1/lstm_hidden_size), size=w_shape), dtype=torch.float32, device = device), requires_grad=True)
        self.b_q = torch.nn.Parameter(torch.tensor(np.random.uniform(-math.sqrt(1/lstm_hidden_size), math.sqrt(1/lstm_hidden_size), size=(vocab_len, )), dtype=torch.float32, device = device), requires_grad=True)                                

        self.softmax = nn.Softmax() # the activation function.


    def forward(self,sentence):
        # print(sentence.shape)
        batch_size = sentence.shape[0]
        batch_input = self.word_embedding_lookup(sentence).cuda()
        # print(batch_input)
        batch_input = batch_input.permute(1, 0, 2)
        # print(batch_input)
        # print(batch_input.size())

        output = ()
        (h_output, cell_output) = init_lstm_state(batch_size, self.lstm_size)
        for word in batch_input:
            # word = word.view(batch_size, -1)
            h_output, cell_output = self.rnn_lstm(word, h_output, cell_output)
            output += (h_output.unsqueeze(0), )
        
        output = torch.cat(output)
        # print(output.shape)

        out = torch.matmul(output, self.W_hq) + self.b_q
        # out = self.softmax(out)
        # print(out.shape)
        # out = torch.argmax(out, dim=-1, keepdim=False) #返回最大值排序的索引值
        output = out.permute(1, 2, 0)   # batch_size, vocab_len, seq_len
        # print(output.shape)
        return {'pred': output.float()}

    def predict(self,sentence):
        with torch.no_grad():
            output = self(sentence)
            predict = torch.argmax(output['pred'], dim=1, keepdim=False)
        return {'pred': predict}
    
    def generate_seq(self, begin_word, end_word, temperature=1.0):
        with torch.no_grad():
            word = begin_word
            predict = []
            (h_output, cell_output) = init_lstm_state(1, self.lstm_size)
            while word != end_word and len(predict) <= 65:
                input = self.word_embedding_lookup(word)
                h_output, cell_output = self.rnn_lstm(input, h_output, cell_output)
                output = torch.matmul(h_output, self.W_hq) + self.b_q
                output = self.softmax(output / temperature)
                word = torch.argmax(output, dim=-1, keepdim=False) #返回最大值排序的索引值
                # print(word)
                predict.append(word.item())

            # out = torch.argmax(out, dim=-1, keepdim=False) #返回最大值排序的索引值
        return predict


# 与torch.nn的LSTM作对比
class RNN_torch_lstm(nn.Module):
    def __init__(self, vocab_len, embedding_size, lstm_hidden_size):
        super(RNN_torch_lstm,self).__init__()
        self.vocab_length = vocab_len
        self.word_embedding_size = embedding_size
        self.lstm_size = lstm_hidden_size
        self.lstm_layer_num = 3 

        # 词向量层，词表大小 * 向量维度
        self.word_embedding_lookup = nn.Embedding(vocab_len, embedding_size).cuda()
        # 网络主要结构
        self.rnn_lstm = nn.LSTM(embedding_size, lstm_hidden_size, self.lstm_layer_num).cuda()
        # 进行分类
        self.linear = nn.Linear(lstm_hidden_size, vocab_len).cuda()                              

        self.softmax = nn.Softmax() # the activation function.


    def forward(self,sentence):
        # print(sentence.shape)
        sentence = sentence.cuda()
        batch_size, seq_len = sentence.shape[0], sentence.shape[1]
        batch_input = self.word_embedding_lookup(sentence)
        # print(batch_input)
        batch_input = batch_input.permute(1, 0, 2)
        # print(batch_input)
        # print(batch_input.size())

        # output = ()
        (h_0, c_0) = (torch.zeros((self.lstm_layer_num, batch_size, self.lstm_size), device = device), 
                                    torch.zeros((self.lstm_layer_num, batch_size, self.lstm_size), device = device))
        # for word in batch_input:
        #     # word = word.view(batch_size, -1)
        #     h_output, cell_output = self.rnn_lstm(word, (h_output, cell_output))
        #     output += (h_output.unsqueeze(0), )
        
        # output = torch.cat(output)
        # print(output.shape)
        output, hidden_and_cell = self.rnn_lstm(batch_input, (h_0, c_0))
        output = self.linear(output)
        # out = torch.matmul(output, self.W_hq) + self.b_q
        # output = self.softmax(output)
        output = output.permute(1, 2, 0)
        # print(output.shape)
        return {'pred': output}

    # def predict(self,sentence):
    #     with torch.no_grad():
    #         output = self(sentence)
    #         predict = torch.argmax(output['pred'], dim=1, keepdim=False)
    #     return {'pred': predict}
    
    def generate_seq(self, begin_word, end_word, temperature=1.0):
        with torch.no_grad():
            word = begin_word
            predict = []
            hidden_and_cell = None
            while word != end_word and len(predict) <= 49:
                input = self.word_embedding_lookup(word.view(1, -1))
                output, hidden_and_cell = self.rnn_lstm(input, hidden_and_cell)
                output = self.linear(output)
                output = self.softmax(output)
                word = torch.argmax(output.view(-1), dim=-1, keepdim=False) #返回最大值排序的索引值
                predict.append(word.item())

        # out = torch.argmax(out, dim=-1, keepdim=False) #返回最大值排序的索引值
        return predict