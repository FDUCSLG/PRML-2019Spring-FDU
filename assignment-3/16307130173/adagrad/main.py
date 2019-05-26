import sys
import numpy as np
import os
import time
import getopt
from torch.utils.data import DataLoader
import torch.optim as opt
from data import *
from model import *
from generate import *

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def train(path):

    start_time = time.clock()
    log_lst = []
    
    device = torch.device("cuda")

    word_dict, full_data = pre_process(path)
    dict_size = len(word_dict)
    full_data = torch.from_numpy(full_data)

    print('Dict Size: ', dict_size)
    
    pick = full_data.size()[0] - Utils.batch_size
    train_data, test_data = full_data[:pick,:], full_data[pick:,:]
    train_loader = DataLoader(train_data, batch_size = Utils.batch_size, shuffle = True, drop_last = True)
    
    model = PoetryModel(dict_size, Utils.embedding_dim, Utils.hidden_dim)
    model.to(device)
    model.load_state_dict(torch.load('model/poem.pth', 'cuda'))
    
    optimizer = opt.Adagrad(model.parameters(), lr = Utils.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(Utils.epoch):
        cnt = 0
        for i, data in enumerate(train_loader):
            cnt += 1
            
            def get_loss(loader):
                data = loader.long().transpose(1, 0).contiguous().to(device)
                x, target = data[:-1, :], data[1:, :]
                output, _ = model(x)
                loss = criterion(output, target.contiguous().view(-1))
                return loss
            
            optimizer.zero_grad()            
            loss = get_loss(data)
            
            loss.backward()
            optimizer.step()

            loss = get_loss(test_data)
            perplexity = torch.mean(torch.exp(loss)).item()
            
            print("[info] epoch: %d, Step: %d, Loss: %0.8f, Perplexity: %0.8f" % (epoch + 1, cnt, loss.item(), perplexity))
            log_lst.append([loss.item(), perplexity])

    torch.save(model.state_dict(), './poem.pth')
    np.savez('log.npz', log=np.array(log_lst))
    print('Total Time Used:', time.clock() - start_time)

if __name__ == '__main__':

    opts, args = getopt.getopt(sys.argv[1:], '', ['train', 'generate', 'path=', 'sentence_size='])

    train_flag = False
    generate_flag = False
    path = ''
    sentence_size = 0
    
    for name, val in opts:
        if name == '--train':
            train_flag = True
        if name == '--path':
            path = val
        if name == '--sentence_size':
            sentence_size = int(val)
        if name == '--generate':
            generate_flag = True

    if path == '':
        print('Please input data path!')
        exit(0)
    if sentence_size == 0 and train_flag == False:
        print('Sentence Size Invalid!')
        exit(0)
        
    if train_flag == True:
        train(path)

    if generate_flag == True:
        device = torch.device("cuda")
        word_dict, train_data = pre_process(path)
        dict_size = len(word_dict)
        
        model = PoetryModel(dict_size, Utils.embedding_dim, Utils.hidden_dim)
        model.to(device)
        model.load_state_dict(torch.load('model/poem.pth', 'cuda'))

        poetry = ''.join(generate(model, '日', word_dict, sentence_size - 1, 4))
        print(poetry)

    exit(0)
