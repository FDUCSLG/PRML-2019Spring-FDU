import import_ipynb
from model import *
from prepareData import *
from torch.utils.data import DataLoader
import torch
import random
import torch.nn.parallel
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda")

def preprocess(path='./json'):
    
    poems = prepareData(path)
    poems = traditional2simplified(poems)
    vocab = prepareVocab(poems)
    fullData = word2idx(poems, vocab)
    fullData = torch.from_numpy(fullData)
    trainSize = int(0.8 * len(fullData))
    testSize = len(fullData) - trainSize
    trainSet, devSet = torch.utils.data.random_split(fullData, [trainSize, testSize])
    return trainSet, devSet, vocab

def perplexity(model, dataset, seqLen=80):
    sumPerp = 0
    for data in dataset:
        data = data.to(device)
        mulProb = 1
        hidden = None
        cell = None
        for i in range(len(data)-1):
            word = data[i]
            predict = data[i+1].item()
            word_id = torch.tensor([word.item()]).view(1,1).cuda()
            output, hidden, cell = model(word_id, hidden, cell)
            prob = F.softmax(output, 1)[0][predict].item()
            mulProb *= prob
        pplxty = pow(1/mulProb, 1/len(data-1))
        sumPerp += pplxty
    return sumPerp/(len(dataset))


def train():
    
    
    trainSet, devSet, vocab = preprocess()
    
    dataLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
    
    model = lstmPoem(len(vocab))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    pplxty = 1000000
    lossSet = []
    pplxSet = []
    for epoch in range(15):
        
        count = 0
        it = 0
        lossSum = 0
        for idx, data in enumerate(dataLoader):
            
            data = data.transpose(1, 0).contiguous()
            #data (seqlen, batch_size)
            data = data.to(device)
            optimizer.zero_grad()
            
            input_, target = data[:-1, :], data[1:, :]
            #input (0:seq-1, batch_size) target(1:seq, batch_size)
            
            output, _, _ = model(input_)
            #output:    (seq * batchsize, vocab)
            
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            lossSum += loss.cpu().detach().numpy()
            print(type(loss))
            count += 1
            print("Epoch: %d batch index:%d loss: %.6f perplexity:%0.8f" % (epoch, idx, loss, torch.mean(torch.exp(loss)).item()))
            gen_poetry = ''.join(generatePoem(model, vocab, "日",20))
            print(gen_poetry)
        lossSet.append(lossSum/count)
        lastPplxty = pplxty
        pplxty = perplexity(model, devSet)
        pplxSet.append(pplxty)
        print("Perplexity after epoch %d : %.6f" % (epoch, pplxty))
#         if pplxty >= lastPplxty:
#             break
        
    return model, lossSet, pplxSet
    #transfer to num

def trainNesterov():
    
    trainSet, devSet, vocab = preprocess()
    
    dataLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
    
    model = lstmPoem(len(vocab))
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    pplxty = 1000000
    lossSet = []
    pplxSet = []
    for epoch in range(15):
        
        count = 0
        it = 0
        lossSum = 0
        for idx, data in enumerate(dataLoader):
            
            data = data.transpose(1, 0).contiguous()
            #data (seqlen, batch_size)
            data = data.to(device)
            optimizer.zero_grad()
            
            input_, target = data[:-1, :], data[1:, :]
            #input (0:seq-1, batch_size) target(1:seq, batch_size)
            
            output, _, _ = model(input_)
            #output:    (seq * batchsize, vocab)
            
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            lossSum += loss.cpu().detach().numpy()
            count += 1
            print("Epoch: %d batch index:%d loss: %.6f perplexity:%0.8f" % (epoch, idx, loss, torch.mean(torch.exp(loss)).item()))
            gen_poetry = ''.join(generatePoem(model, vocab, "日",20))
            print(gen_poetry)
        lossSet.append(lossSum/count)
        lastPplxty = pplxty
        pplxty = perplexity(model, devSet)
        pplxSet.append(pplxty)
        print("Perplexity after epoch %d : %.6f" % (epoch, pplxty))
#         if pplxty >= lastPplxty:
#             break
        
    return model, lossSet, pplxSet
    #transfer to num

def generatePoem(model, vocab, start_word, length):
    start_id = torch.tensor([vocab.to_index(start_word)]).view(1,1).cuda()
    result = []
    result.append(start_word)
    hidden = None
    cell = None
    input_ = start_id
    for i in range(length-1):
        
        output, hidden, cell = model(input_, hidden, cell)
        
        top10 = output[0].topk(10)
#         print(output)
        
        index = top10[1][random.randint(0,9)].item()
        w = vocab.to_word(index)
        result.append(w)
        input_ = torch.tensor([index]).view(1,1).cuda()
        
    return result

def drawPic(y):
    plt.figure()
    y = y.numpy().reshape(-1)
    x = np.arange(1,len(y)+1,1)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

model = train()

import matplotlib.pyplot as plt
def drawPic(y):
    plt.figure()
    y = y.reshape(-1)
    x = np.arange(1,len(y)+1,1)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.show()

drawPic(np.array(model[2]))
drawPic(np.array(model[1])

