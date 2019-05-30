import os
os.sys.path.append('..')
import json
import re
import math
import torch.nn as nn
import torch
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
import torch.autograd as autograd
import torch.optim as optim
from enum import IntEnum
import random

from typing import *
import matplotlib.pyplot as plt


# ====================== data process =======================

def read_file(num = 4, senten_len=50):
  # get a poetry list 
  basePath = './dataset/poet.tang.'
  dataset = ""
  count_words = {}
  for i in range(num):
    path = basePath + str(i*1000) + '.json'
    with open(path, 'r', encoding='utf-8') as f:
      data = json.loads(f.read())
    for item in data:
      temp = re.sub(r'（.*?）', "", "".join(item['paragraphs']))
      temp = re.sub(r'\[.*?\]', "", temp)
      temp = re.sub(r'「.*?」', "", temp)
      temp = re.sub(r'□', "", temp)
      temp = re.sub(r'{.*?}', "", temp)
      temp_len = len(temp)
      if temp_len > 0 and temp_len < senten_len:
        for word in temp:
          if word in count_words:
            count_words[word] += 1
          else:
            count_words[word] = 1
        dataset += temp + "@"
  # get rid of word with low frequency 
  erase = []
  for key in count_words:
    if count_words[key] <= 1:
      erase.append(key)
  for key in erase:
    del count_words[key]
  wordPairs = sorted(count_words.items(), key=lambda x: -x[1])
  words, _ = zip(*wordPairs)
  words += ("$", "#", "-",)
  # word to id
  word2num = dict((c, i) for i, c in enumerate(words))
  num2word = dict((i, c) for i, c in enumerate(words))
  return word2num, num2word, dataset

def get_batch_data(num = 8, batch_s=256, senten_len=50):
  word2num, num2word, dataset = read_file(num, senten_len)
  word2numF = lambda char: word2num.get(char, word2num["#"])
  dataset = dataset.split("@")
  dataset.pop()
  data_len = len(dataset)
  # get one-hot
  handle_data = lambda item: list(map(word2numF, item[:len(item)-1])) + [word2numF("$")] + [word2numF("-")]*(senten_len-len(item)) 
  handle_target = lambda item: list(map(word2numF, item[1:])) + [word2numF("$")] + [word2numF("-")]*(senten_len-len(item))
  data = torch.LongTensor([handle_data(item) for item in dataset]) 
  target = torch.LongTensor([handle_target(item) for item in dataset]) 
  # divide dataset into train and test
  train_data_batch, train_target_batch, test_data_batch, test_target_batch = [], [], [], []
  train_batch_len = math.floor(data_len*0.9 / batch_s)
  t = 0
  while t*batch_s < data_len:
    if t < train_batch_len:
      train_data_batch.append(data[t*batch_s : (t+1)*batch_s])
      train_target_batch.append(target[t*batch_s : (t+1)*batch_s])
    elif (t+1)*batch_s < data_len:
      test_data_batch.append(data[t*batch_s : (t+1)*batch_s])
      test_target_batch.append(target[t*batch_s : (t+1)*batch_s])
    else:
      test_data_batch.append(data[t*batch_s:])
      test_target_batch.append(target[t*batch_s:])
    t += 1
  train_data = (dataset[:train_batch_len*batch_s], train_data_batch, train_target_batch)
  test_data = (dataset[train_batch_len*batch_s:], test_data_batch, test_target_batch)
  return word2num, num2word, train_data, test_data


# ====================== model definition ======================

class Dim(IntEnum):
  # assumes input matrix is of shape (batch, seqence, feature)
  batch = 0
  seq = 1
  feature = 2

class LSTMLayer(nn.Module):
  def __init__(self, data_dim: int, hidden_dim: int):
    super().__init__()
    self.data_dim = data_dim
    self.hidden_dim = hidden_dim
    self.W_h = Parameter(torch.Tensor(hidden_dim, hidden_dim*4)) 
    self.w_x = Parameter(torch.Tensor(data_dim, hidden_dim*4))
    self.bias = Parameter(torch.Tensor(hidden_dim*4))
    self._init_weights()

  def _init_weights(self):
    for p in self.parameters():
      if p.data.ndimension() >= 2:
        nn.init.xavier_uniform_(p.data)
      else:
        nn.init.zeros_(p.data)

  def forward(self, x, _init_states=None):
    batch_s, seq_s, _ = x.size()
    hidden_seq = []
    if _init_states is None:
      h_t, c_t = torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)
    else:
      h_t, c_t = _init_states

    HD = self.hidden_dim
    for t in range(seq_s):
      x_t = x[:, t, :]
      gates = h_t@self.W_h + x_t@self.w_x + self.bias
      f_t = torch.sigmoid(gates[:, :HD])
      i_t = torch.sigmoid(gates[:, HD:HD*2])
      c_h_t = torch.tanh(gates[:, HD*2:HD*3])
      o_t = torch.sigmoid(gates[:, HD*3:])
      c_t = f_t * c_t + i_t * c_h_t
      h_t = o_t * torch.tanh(c_t)
      hidden_seq.append(h_t.unsqueeze(Dim.batch))
    hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
    hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
    return hidden_seq, (h_t, c_t)


class MyModel(nn.Module):
  def __init__(self, feat_s: int, hidden_dim: int, dict_len: int):
    super().__init__()
    self.feat_s = feat_s
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(dict_len, feat_s)
    # self.lstm = nn.LSTM(input_size=feat_s, hidden_size=hidden_dim, batch_first=True)
    self.lstm = LSTMLayer(feat_s, hidden_dim)
    self.linear = nn.Linear(hidden_dim, dict_len)
  
  def forward(self, data_batch, _init_states=None):  # data_batch: [batch_s, senten_len]
    # embedding  input_batch: [batch_s, senten_len, feat_s]
    input_batch = self.embedding(data_batch)
    if _init_states is None:
      output_batch, states = self.lstm(input_batch)
    else:
      output_batch, states = self.lstm(input_batch, _init_states)
    output_batch = self.linear(output_batch)
    return output_batch, states


# ====================== parameters definition ======================

batch_s, senten_len, feat_s, hidden_dim = 256, 50, 128, 256
word2num, num2word, train_data, test_data = get_batch_data()
dict_len = len(word2num)
word2numF = lambda char: word2num.get(char, word2num["#"])
num2wordF = lambda num: num2word.get(num)


model = MyModel(feat_s, hidden_dim, dict_len)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
optimizer_a = optim.Adagrad(model.parameters())
optimizer_b = optim.Adadelta(model.parameters(), lr=0.5)
softmax_d0 = nn.Softmax(dim=0)
softmax_d1 = nn.Softmax(dim=1)
softmax_d2 = nn.Softmax(dim=2)

test_loss = []
test_perplexity = []

# ======================== utils function =========================

def translation(predict_poetry):
  poetry_vec = torch.max(softmax_d1(predict_poetry), dim=1)[1].tolist()
  poetry = ""
  for item in poetry_vec:
    word = num2wordF(item)
    poetry += word
  return poetry, poetry_vec

def print_example(num, batch_len, poetry, begin, predict_poetry):
  for i in range(num):
    print("-------------------------------------------\n")
    j = np.random.randint(0, batch_len)
    print("原句：", poetry[begin+j], "\n")
    print("预测：", translation(predict_poetry[j])[0], "\n")

def get_perplexity(output, target):
  batch_len, _, _ = output.shape
  y = target.view(-1).tolist()
  y_h = torch.log(softmax_d2(output).view(-1, dict_len), out=None)
  perplexity = 0
  for i in range(len(y)):
    perplexity += (y_h[i][y[i]]).item()
  return perplexity / (senten_len*batch_len)

def print_paras(paras):
  for i in paras:
    print(i.data.shape)


# ========================= train model =========================

def train():
  model.train()
  model.zero_grad()
  train_poetry, train_data_batch, train_target_batch = train_data
  train_batch_len = len(train_data_batch)
  for i in range(train_batch_len):
    optimizer.zero_grad()
    output, _ = model(train_data_batch[i])
    # print_paras(model.parameters())
    loss = loss_func(output.view(-1, dict_len), train_target_batch[i].view(-1))
    # print_example(3, train_data_batch[i].shape[0], train_poetry, i*batch_s, output)
    print("LOSS: ", loss.item())
    loss.backward()
    optimizer.step()
    # optimizer_a.step()
    # optimizer_b.step()

def test():
  model.eval()
  test_poetry, test_data_batch, test_target_batch = test_data
  loss, perplexity, test_batch_len = 0, 0, len(test_data_batch)
  for i in range(test_batch_len):
    output, _ = model(test_data_batch[i])
    loss += loss_func(output.view(-1, dict_len), test_target_batch[i].view(-1)).item()
    perplexity += get_perplexity(output, test_target_batch[i])
    # print_example(5, test_data_batch[i].shape[0], test_poetry, i*batch_s, output)
  loss /= test_batch_len
  perplexity = np.exp2(perplexity/-test_batch_len)
  test_loss.append(loss)
  test_perplexity.append(perplexity)
  if len(test_loss)>1 and test_loss[-2]<loss:
    return True
  print("========================= LOSS: ", loss, "\n")
  print("========================= perplexity", perplexity, "\n")


def predict(first_char):
  model.eval()
  data_batch = torch.LongTensor([word2numF(first_char)]).unsqueeze(0)
  prediction = first_char
  states = (torch.zeros(hidden_dim), torch.zeros(hidden_dim))
  char_list = []
  # states = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
  for i in range(senten_len):
    output, states = model(data_batch, states)
    next_char, next_char_num = translation(output[0])
    prediction += next_char
    # if i % 12 == 4:
    #   k = 0
    # elif i % 12 == 10:
    #   k = 1
    # else:
    #   k = np.random.randint(2, dict_len/50)
    # char_list.append(num2wordF(k))
    data_batch = torch.LongTensor(next_char_num).unsqueeze(0)
  # print(char_list)
  return prediction


def train_model(times=30):
  over_time = times
  for i in range(times):
    print("=======================================", i+1, "\n")
    train()
    if test():
      over_time = i+1
      break
  print(predict('日'), "\n")
  print(predict('紅'), "\n")
  print(predict('山'), "\n")
  print(predict('夜'), "\n")
  print(predict('湖'), "\n")
  print(predict('海'), "\n")
  print(predict('月'), "\n")
  plt.ylabel("CROSS ENTROPY")
  plt.title("CROSS ENTROPY CHANGE")
  plt.plot([i+1 for i in range(over_time)], test_loss)
  plt.show()
  plt.ylabel("PERPLEXITY")
  plt.title("PERPLEXITY CHANGE")
  plt.plot([i+1 for i in range(over_time)], test_perplexity)
  plt.show()

train_model()





# def train_a():
#   lstm_a.train()
#   lstm_a.zero_grad()
#   for i in range(len(train_input_batch)):
#     item = train_input_batch[i]
#     optimizer_a.zero_grad()
#     y, _ = lstm_a(autograd.Variable(item))
#     y = linear_a(y)
#     print(list(linear_a.parameters()))
#     # for j in range(3):
#     #   print("--------------------------\n")
#     #   k = random.randint(0, len(train_input_batch))
#     #   print("原句：", train_dataset[i*batch_s+k], "\n")
#     #   print("预测：", translation(y[k]), "\n")
#     loss = loss_func(y.view(-1, dict_len), train_target_batch[i].view(-1))
#     print("LOSS：", loss.item())
#     loss.backward()
#     optimizer_a.step()



# def test_a():
#   lstm_a.eval()
#   loss = 0
#   perplexity = 0
#   test_input_len = len(test_input_batch)
#   for i in range(test_input_len):
#     item = test_input_batch[i]
#     y, _ = lstm_a(item)
#     y = linear_a(y)
#     # for j in range(5):
#     #   print("-------------------------------\n")
#     #   k = random.randint(0, len(test_input_batch))
#     #   print("原句：", test_dataset[i*batch_s+k], "\n")
#     #   print("预测：", translation(y[k]), "\n")
#     loss += (loss_func(y.view(-1, dict_len), test_target_batch[i].view(-1))).item()
#     perplexity += get_perplexity(y, test_target_batch[i])
#   loss /= test_input_len
#   perplexity = np.exp2(perplexity/-test_input_len)
#   test_loss.append(loss)
#   test_perplexity.append(perplexity)
#   print("========================= LOSS: ", loss, "\n")
#   print("========================= perplexity", perplexity, "\n")





# def predict_a(first_char):
#   lstm_a.eval()
#   tensor = word_vectors[word2num.get(first_char, word2num["#"])].unsqueeze(0).unsqueeze(0)
#   prediction = first_char
#   paras = lstm_a.parameters()
#   for p in paras:
#     print(p.data)
#   # parameter initialization
#   para = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
#   for i in range(senten_len):
#     y, para = lstm_a(tensor, para)
#     y = softmax_d0(linear_a(y).view(-1))
#     next_char_num = torch.max(y, dim=0)[1].item()
#     tensor = ((word_vectors[next_char_num]).unsqueeze(0)).unsqueeze(0)
#     word = num2wordF(next_char_num)
#     prediction += word
#   return prediction
    
# def train_model(times=50):
#   for i in range(times):
#     train_a()
#     test_a()
#     print("******************", i)
#     print(predict_a('日'), "\n")
#     print(predict_a('春'), "\n")
#     # print(predict_a('山'), "\n")
#     # print(predict_a('夜'), "\n")
#     # print(predict_a('湖'), "\n")
#     # print(predict_a('海'), "\n")
#     # print(predict_a('月'), "\n")
#   print("LOSS:  ", test_loss)
#   plt.plot([i+1 for i in range(times)], test_loss)
#   plt.show()
#   print("PERPLEXITY:   ", test_perplexity)
#   plt.plot([i+1 for i in range(times)], test_perplexity)
#   plt.show()

# train_model()
