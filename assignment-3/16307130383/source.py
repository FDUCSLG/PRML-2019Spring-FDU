import json
import string
import random
import os
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

def load_json():
  # this finds our json files
  path_to_json = '../json'
  global json_files
  json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
  data = []
  # we need both the json and an index number so use enumerate()
  for index, js in enumerate(json_files):
      with open(os.path.join(path_to_json, js)) as json_file:
          data += json.load(json_file)
  paragraphs = [item['paragraphs'] for item in data]
  random.Random(1).shuffle(paragraphs)
  # write to disk
  with open('paragraphs.list', 'wb') as fp:
    pickle.dump(paragraphs, fp)

paragraphs = []
if os.path.exists('paragraphs.list'):
  with open ('paragraphs.list', 'rb') as fp:
    paragraphs = pickle.load(fp)
else:
  load_json()

#####################
# Preprocess
#####################
def get_vocab_seqs(sample_size=1000, seq_lenth=48):
  assert sample_size <= len(paragraphs)
  words = []
  count = {}
  vocab = {}
  sequences = []
  if os.path.exists(str(sample_size)+'words.list'):
    with open(str(sample_size)+'words.list', 'rb') as fp:
      words = pickle.load(fp)
    with open(str(sample_size)+'count.dict', 'rb') as fp:
      count = pickle.load(fp)
    with open(str(sample_size)+'vocab.dict', 'rb') as fp:
      vocab = pickle.load(fp)
    with open(str(sample_size)+'sequences.list', 'rb') as fp:
      sequences = pickle.load(fp)
  else:
    # create vocab
    words = ['EOS', 'OOV']
    count = {'EOS': 100000, 'OOV': 100000}
    for paragraph in paragraphs[:sample_size]:
      for sentence in paragraph:
        for word in sentence:
          words.append(word)
          if word in count:
            count[word] += 1
          else:
            count[word] = 0
    words = set([ x for x in words if count[x] > 0 ])  # at least one time
    words = list(words)
    vocab = {word: i for i, word in enumerate(words)}
    # cut into fix length sentence
    sequences = []
    for paragraph in paragraphs[:sample_size]:
      seq = ""
      for sentence in paragraph:
        seq += sentence
        if len(seq) >= seq_lenth:
          sequences.append(seq[:seq_lenth])
          seq = ""
    # write to disk
    with open(str(sample_size)+'words.list', 'wb') as fp:
      pickle.dump(words, fp)
    with open(str(sample_size)+'count.dict', 'wb') as fp:
      pickle.dump(count, fp)
    with open(str(sample_size)+'vocab.dict', 'wb') as fp:
      pickle.dump(vocab, fp)
    with open(str(sample_size)+'sequences.list', 'wb') as fp:
      pickle.dump(sequences, fp)
  # seq to int
  xs, ys = seq_to_int(vocab, sequences)
  return count, words, vocab, sequences, xs, ys

def seq_to_int(vocab, sequences):
  xs = []
  ys = []
  for sequence in sequences:
    seq = []
    for word in sequence:
      if word in vocab:
        seq.append(vocab[word])
      else:
        seq.append(vocab['OOV'])
    seq_next = seq[1:]
    seq_next.append(vocab['EOS'])
    xs.append(seq)
    ys.append(seq_next)
  return xs, ys

#####################
# LSTM model
#####################
class LSTM(nn.Module):
  def __init__(self, vocab_size, input_dim, hidden_dim, batch_size, output_dim=1,
                  num_layers=2):
    super(LSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.num_layers = num_layers
    # Embedding fn
    self.embed = nn.Embedding(vocab_size, self.input_dim)
    # Define the LSTM layer
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
    # Define the output layer
    self.linear = nn.Linear(self.hidden_dim, output_dim)

  def init_hidden(self):
    # return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #         torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

  def forward(self, input):
    # embed input
    input = self.embed(input)
    # Forward pass
    lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))  # lstm_out shape [input_size, batch_size, hidden_dim]
    # linear to dim-V, then softmax
    y_pred = self.linear(lstm_out.view(len(input), self.batch_size, -1))  # shape [seq_len, batch_size, output_len]
    # softmax
    # don't have to, cause CE already have
    # y_pred = nn.functional.softmax(y_pred, dim=2)
    return y_pred

#####################
# Init
#####################
seq_len = 48  # 句长
sample_size = 2048  # 诗个数
word_count, words, vocab, sequences, X_int, y_int = get_vocab_seqs(sample_size, seq_len)
vocab_size = len(words)
input_dim = 256  # 输入向量长
hidden_dim = 512
batch_size = 32
output_dim = vocab_size
num_layers = 1
model = LSTM(vocab_size=vocab_size, input_dim=input_dim, hidden_dim=hidden_dim, 
        batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
# loss_fn & optimiser 
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.01
# claim globally
X_train_array = []
target_array = []
hist = []
# preprocess X_train, y_train
for batch_start in range(0, len(X_int) - batch_size, batch_size):
  l = batch_start
  r = batch_start + batch_size
  X_transpose = np.array(X_int[l:r]).transpose()
  X_train = torch.from_numpy(X_transpose).type(torch.LongTensor)
  y_transpose = np.array(y_int[l:r]).transpose()
  y_train = torch.from_numpy(y_transpose).type(torch.LongTensor)
  target = y_train.contiguous().view(-1)
  X_train_array.append(X_train)
  target_array.append(target)

def get_var():
  return {'X': X_train_array, 'y': target_array, 'V': vocab, 'seq': sequences, 'm': model, 'w': words, 'h': hist, 'c': word_count}

#####################
# Trainer
#####################
def train(num_epochs, batch_end=len(X_train_array), lr=learning_rate, print_rate=8):
  global hist
  hist = np.zeros(num_epochs)
  # adjust learning rate
  optimiser = torch.optim.Adam(model.parameters(), lr=lr)
  # optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  for t in range(num_epochs):
    # get batch
    global X_train_array
    global target_array
    batch_num = t % batch_end
    X_train = X_train_array[batch_num]
    target = target_array[batch_num]
    # Clear stored gradient
    model.zero_grad()
    # Initialise hidden state
    model.hidden = model.init_hidden()
    # Forward pass
    y_pred = model(X_train)
    # reshape output
    output = y_pred.view(-1, vocab_size)
    # computeloss
    loss = loss_fn(output, target)
    hist[t] = loss.item()
    # print
    if t % print_rate == 0:
      print("batch num: ", batch_num)
      print("Epoch ", t, "CE: ", loss.item())
      print("sample input: ", translate_int( X_train.numpy().transpose().tolist()[0] ))
      print("sample output: ", translate_vec( y_pred.detach().permute(1, 0, 2).numpy().tolist()[0] ))
    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()
    # backward
    loss.backward()
    # Update parameters
    optimiser.step()

#####################
# Generate
#####################
def generate(seed=None, length=48, temperature=0.8):
  # set up first char
  start = [[ random.randint(0, vocab_size-1) ]]
  if not seed is None:
    if seed in words:
      start = [[ vocab[seed] ]]
      # print( vocab[seed] )
    else:
      start = [[ vocab['OOV'] ]]
  start = torch.LongTensor(start)
  # init hidden
  model.hidden = model.init_hidden()
  # generate
  seq = ""
  seq += translate_int([ start ])
  for i in range(length):
    start = model.embed(start)
    lstm_out, model.hidden = model.lstm(start.view(1, 1, -1))  # seq_len=1, batch_size=1
    y_pred = model.linear(lstm_out.view(1, 1, -1))
    # according to temperature softmax
    predict = y_pred.data.view(-1).div(temperature).exp()
    predict = torch.multinomial(predict, 1)[0]
    # get next
    next_char = translate_int([ predict ])
    seq += next_char
    # use predict as start
    start = predict
  print(seq)

#####################
# Utilities
#####################
def translate_int(sentence):  # only accept list
  # print('tranlate: ', sentence)
  seq = ""
  for idx in sentence:
    seq += words[idx]
  return seq

def translate_vec(sentence):  # only accept list
  seq = ""
  for vec in sentence:
    idx = np.argmax(vec)
    seq += words[idx]
  return seq

def save(path):
  torch.save(model.state_dict(), path)

def load(path):
  model.load_state_dict(torch.load(path))
  model.eval()

def draw():
  X = [i for i in hist if i != 0]
  line_x = np.linspace(0, len(X), len(X))
  line_y = np.array(X)
  plt.plot(line_x, line_y)
  plt.show()