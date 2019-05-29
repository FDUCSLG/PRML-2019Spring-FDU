import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Sampler
import numpy as np
import argparse
import time
from models import *
from prepare_data import *

data_path = "data.txt"
test_data, train_data = import_data(data_path)
test_data, train_data, idx2word, word2idx = construct_vec(
    test_data, train_data)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--seq_len", "-sl", type=int, default=40)
parser.add_argument("--seq_step", "-ss", type=int, default=10)
parser.add_argument("--hidden_size", "-hs", type=int, default=200)
parser.add_argument("--input_size", "-is", type=int, default=200)
parser.add_argument("--max_epoch", "-me", type=int, default=20)
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
parser.add_argument("--optimizer", "-o", type=str,
                    default="Adam", choices=["Adam", "SGD"])
parser.add_argument("--momentum", "-m", type=float, default=0.9)
parser.add_argument("--tolerance", "-t", type=int, default=2)
arg = parser.parse_args()
batch_size = arg.batch_size
seq_length = arg.seq_len
seq_step = arg.seq_step
hidden_size = arg.hidden_size
input_size = arg.input_size
max_epoch = arg.max_epoch
lr = arg.learning_rate
train_data = PoemSet(train_data, seq_length, seq_step, idx2word, word2idx)
test_data = PoemSet(test_data, seq_length, seq_step, idx2word, word2idx)
print(len(train_data), len(test_data))
dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)


def train():
    net = TangPoemGenerator(input_size, hidden_size, len(idx2word))
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    if arg.optimizer == "Adam":
        optimizer = Adam(net.parameters(recurse=True), lr=lr)
    elif arg.optimizer == "SGD":
        optimizer = SGD(net.parameters(recurse=True), lr=lr,momentum=arg.momentum)
    p = []
    for epoch in range(max_epoch):
        net.train()
        losses = 0
        for idx, batch_data in enumerate(dataloader):
            t0 = time.time()
            batch_data = batch_data.transpose(1, 0).contiguous()
            if use_cuda:
                batch_data = batch_data.cuda()
            input, target = batch_data[:-1, :], batch_data[1:, :]
            output = net(input)
            optimizer.zero_grad()
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            t1 = time.time()
            if idx % 10 == 0:
                print("epoch:", epoch, " step:", idx,
                      " loss:", losses/(idx+1), " time:", t1-t0)
        torch.save(net.state_dict(), '%s_%s.pth' %
                   ("Tang", epoch))
        perplexity = eval(net)
        print("perplexity:", perplexity)
        p.append(perplexity)
        if epoch >= arg.tolerance and perplexity > p[epoch-arg.tolerance]:
            print("early stop")
            break


def eval(net):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    perplexity = 0
    use_cuda = torch.cuda.is_available()
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            data = data.transpose(1, 0).contiguous()
            if use_cuda:
                data = data.cuda()
            input, target = data[:-1, :], data[1:, :]
            output = net(input)
            loss = criterion(output, target.view(-1))
            p = torch.exp(loss).item()
            perplexity += p/len(test_dataloader)
            print("testing %d/%d,perplexity:%f" %
                  (idx, len(test_dataloader), p))
    return perplexity


train()
