import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import time
import os
import numpy as np

from config import Config
from model import PoetryModel
from generate import *

config = Config()

def train():
    datas = np.load("tang.npz")
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data[:5000],
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=2)

    model = PoetryModel(len(word2ix),
                        embedding_dim=config.embedding_dim,
                        hidden_dim = config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.to(config.device)


    f = open('result.txt','w')
    loss_history = []
    for epoch in range(config.epoch):
        start_time = time.time()
        temp_loss = 0

        for step, batch_data in enumerate(dataloader):
            batch_data = batch_data.long().transpose(1,0).contiguous()
            optimizer.zero_grad()
            trn, target = batch_data[:-1,:],batch_data[1:,:]
            output, _ = model(trn)
            loss = criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            if step % config.print_freq == 0 or step == len(dataloader) - 1:
                print(
                    "Train: [{:2d}/{}] Step: {:03d}/{:03d} Loss: {} ".format(
                        epoch + 1, config.epoch, step, len(dataloader) - 1, loss.item()))

        loss_history.append(temp_loss / len(len(dataloader)))
        elapsed_time = time.time() - start_time
        print("Epoch: %d" % epoch + " " + "Loss: %d" % loss_history[-1] + " Epoch time: " + time.strftime(
            "%H: %M: %S", time.gmtime(elapsed_time)))
        torch.save(model.state_dict(),config.model_path)


def test():
    datas = np.load("tang.npz")
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    model = PoetryModel(len(ix2word), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(config.model_path, config.device))
    while True:
        start_words = str(input())
        gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix, config))
        print(gen_poetry)


if __name__ == '__main__':
    if os.path.exists(config.model_path):
        test()
    else:
        train()
        test()
