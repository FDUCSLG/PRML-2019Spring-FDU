import json
import string
import time
import random
import os
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
"""
def load_json():
    path_to_json = '../json'
    global json_files
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    data = []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            data += json.load(json_file)
    paragraphs = [item['paragraphs'] for item in data]
    random.Random(1).shuffle(paragraphs)
    with open('paragraphs.list', 'wb') as fp:
        pickle.dump(paragraphs, fp)
    return paragraphs

paragraphs = []
if os.path.exists('paragraphs.list'):
    with open ('paragraphs.list', 'rb') as fp:
        paragraphs = pickle.load(fp)
else:
    paragraphs = load_json()







# Preprocess
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
        words = set([ x for x in words if count[x] > 0 ])
        words = list(words)
        vocab = {word: i for i, word in enumerate(words)}
        sequences = para_to_seq(paragraphs, sample_size, seq_lenth)
        with open(str(sample_size)+'words.list', 'wb') as fp:
            pickle.dump(words, fp)
        with open(str(sample_size)+'count.dict', 'wb') as fp:
            pickle.dump(count, fp)
        with open(str(sample_size)+'vocab.dict', 'wb') as fp:
            pickle.dump(vocab, fp)
        with open(str(sample_size)+'sequences.list', 'wb') as fp:
            pickle.dump(sequences, fp)
    xs, ys = seq_to_int(vocab, sequences)
    return count, words, vocab, sequences, xs, ys

def para_to_seq(paragraphs, sample_size, seq_lenth):
    sequences = []
    for paragraph in paragraphs[:sample_size]:
        seq = ""
        for sentence in paragraph:
            seq += sentence
            if len(seq) >= seq_lenth:
                sequences.append(seq[:seq_lenth])
                seq = ""
    return sequences

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






# LSTM model

class LSTMcell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_ii = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hi = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = Parameter(torch.Tensor(hidden_dim))
        self.W_if = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hf = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = Parameter(torch.Tensor(hidden_dim))
        self.W_ig = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hg = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = Parameter(torch.Tensor(hidden_dim))
        self.W_io = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ho = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = Parameter(torch.Tensor(hidden_dim))
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def forward(self, input, init_states):
        seq_len = len(input)
        hidden_seq = []
        h_t, c_t = init_states
        for t in range(seq_len):
            x_t = input[t]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq).view(seq_len, -1, hidden_dim)
        return hidden_seq, (h_t, c_t)

class LSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, self.input_dim)
        self.lstm = LSTMcell(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def init_hidden(self, batch_size=32):
        self.hidden = (torch.zeros(batch_size, self.hidden_dim),torch.zeros(batch_size, self.hidden_dim))
                
    def forward(self, input, batch_size=32):
        input = self.embed(input)
        lstm_out, self.hidden = self.lstm(input.view(len(input), batch_size, -1), self.hidden)
        y_pred = self.linear(lstm_out.view(len(input), batch_size, -1))
        return y_pred







# Init

seq_len = 48
sample_size = 16384
word_count, words, vocab, sequences, X_int, y_int = get_vocab_seqs(sample_size, seq_len)
vocab_size = len(words)
input_dim = 256  #
hidden_dim = 512
batch_size = 32
output_dim = vocab_size
num_layers = 1
model = LSTM(vocab_size=vocab_size, input_dim=input_dim, hidden_dim=hidden_dim,
             batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.01
loss_array = []
perplexity_array = []

def int_to_train_target(X_int, y_int, batch_size):
    X_train_array = []
    y_train_array = []
    target_array = []
    for batch_start in range(0, len(X_int) - batch_size, batch_size):
        l = batch_start
        r = batch_start + batch_size
        X_transpose = np.array(X_int[l:r]).transpose()
        X_train = torch.from_numpy(X_transpose).type(torch.LongTensor)
        X_train_array.append(X_train)
        y_transpose = np.array(y_int[l:r]).transpose()
        y_train = torch.from_numpy(y_transpose).type(torch.LongTensor)
        y_train_array.append(y_train)
        target = y_train.contiguous().view(-1)
        target_array.append(target)
    return X_train_array, y_train_array, target_array

X_train_array, y_train_array, target_array = int_to_train_target(X_int, y_int, batch_size)

def get_var():
    return {'X': X_train_array, 'y': target_array, 'V': vocab, 'seq': sequences, 'm': model, 'w': words, 'h': loss_array, 'c': word_count}







# Trainer
def train(num_epochs, batch_end=len(X_train_array), lr=learning_rate, print_rate=8):
    global loss_array
    loss_array = np.zeros(num_epochs)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(num_epochs):
        global X_train_array
        global target_array
        batch_num = t % batch_end
        X_train = X_train_array[batch_num]
        target = target_array[batch_num]
        model.zero_grad()
        model.init_hidden()
        y_pred = model(X_train)
        output = y_pred.view(-1, vocab_size)
        loss = loss_fn(output, target)
        loss_array[t] = loss.item()
        if t % print_rate == 0:
            print("batch num: ", batch_num)
            print("Epoch ", t, "CE: ", loss.item())
            print("sample input: ", translate_int( X_train.numpy().transpose().tolist()[0] ))
            print("sample output: ", translate_vec( y_pred.detach().permute(1, 0, 2).numpy().tolist()[0] ))
        if batch_num == 0:
            perplexity_array.append(test().detach().numpy().tolist())
            save( str(sample_size)+'-'+str( int(t/batch_end) )+'-'+time.strftime("%X", time.localtime())+'.model' )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()






# Test
X_test_array = []
y_test_array = []
target_test_array = []
def test():
    perplexity_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    global X_test_array
    global y_test_array
    global target_test_array
    if X_test_array == []:
        sequences = para_to_seq(paragraphs[sample_size:], int(sample_size / 6), seq_len)
        X_int, y_int = seq_to_int(vocab, sequences)
        X_test_array, y_test_array, target_test_array = int_to_train_target(X_int, y_int, 1)
    perp = 0
    for i in range(len(X_test_array)):
        X_test = X_test_array[i]
        y_test = y_test_array[i]
        model.init_hidden(batch_size=1)
        output = model(X_test, batch_size=1)
        perp_one_batch = perplexity_fn(output.view(-1, vocab_size), target_test_array[i])
        perp += perp_one_batch
        if i % 256 == 0:
            print("Batch Num: ", i, "  Perp: ", perp_one_batch)
            print("sample input: ", translate_int( X_test.detach().permute(1, 0).numpy().tolist()[0] ))
            print("sample output: ", translate_vec( output.detach().permute(1, 0, 2).numpy().tolist()[0] ))
    return perp / len(X_test_array)

def perplexity(output, target):
    output = output.permute(1, 0, 2)
    target = target.permute(1, 0)
    prod_sum = 0
    perps = []
    for y_pred, y in zip(output, target):
        prod = np.sum(np.log(list(map(lambda x: prob_in_pred(y, x), enumerate(y_pred)))))
        prod = min(1000.0, prod)
        prod_sum += prod
        perps.append(prod)
    return prod_sum / len(output)

def prob_in_pred(y, x):
    i, vec = x
    idx = y[i].type(torch.LongTensor)
    return 1.0 / vec[idx]








# Generate
def generate(seed=None, length=48, temperature=0.8, split_len=6):
    start = [[ random.randint(0, vocab_size-1) ]]
    if not seed is None:
        if seed in words:
            start = [[ vocab[seed] ]]
        else:
            start = [[ vocab['OOV'] ]]
    start = torch.LongTensor(start)
    model.init_hidden(batch_size=1)
    seq = ""
    seq += translate_int([ start ])
    for i in range(length):
        y_pred = model(start, batch_size=1)
        predict = y_pred.data.view(-1).div(temperature).exp()
        predict = torch.multinomial(predict, 1)[0]
        if (i + 2) % (split_len * 2) == 0:
            predict = torch.LongTensor([[ vocab['。'] ]])
        elif (i + 2) % split_len == 0:
            predict = torch.LongTensor([[ vocab['，'] ]])
        next_char = translate_int([ predict ])
        seq += next_char
        start = predict
    print(seq)


# Utilities
def translate_int(sentence):
    seq = ""
    for idx in sentence:
        seq += words[idx]
    return seq

def translate_vec(sentence):
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
    print(perplexity_array)
    plt.subplot(2, 1, 1)
    X = [i for i in loss_array if i != 0]
    line_x = np.linspace(0, len(X), len(X))
    line_y = np.array(X)
    plt.plot(line_x, line_y)
    plt.subplot(2, 1, 2)
    X = perplexity_array
    line_x = np.linspace(0, len(X), len(X))
    line_y = np.array(X)
    plt.plot(line_x, line_y)
    plt.show()
"""
