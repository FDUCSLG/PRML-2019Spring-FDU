import import_ipynb
from model import *
from prepareData import *
from torch.utils.data import DataLoader
import torch
import random
import torch.nn.parallel
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

class GRUPoem(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super(GRUPoem, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        input_dim = hidden_dim + embedding_dim
        
        #init weight
        self.Wr = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(input_dim + hidden_dim)))
        self.Br = Parameter(torch.rand(hidden_dim, 1))
        self.Wz = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(hidden_dim + hidden_dim)))
        self.Bz = Parameter(torch.rand(hidden_dim, 1))
        self.Wh = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(input_dim + hidden_dim)))
        self.Bh = Parameter(torch.rand(hidden_dim, 1))
        self.W = Parameter(torch.rand(vocab_size, hidden_dim) * np.sqrt(2/(vocab_size + hidden_dim)))
        self.b = Parameter(torch.rand(vocab_size, 1))

    def forward(self, x, hidden=None):
#         x:    seq_len * batch_size
        seq_len, batch_size = x.size()
    
        if hidden is None:
            Ht = x.data.new(self.hidden_dim, batch_size).fill_(0).float()
        else:
            Ht = hidden
            
        
        embeds = self.embedding(x)
        # seq * batch * embedding
        
        output= []
        
        
        for i in range(len(embeds)):
            
            xTmp = embeds[i].transpose(1,0).contiguous()
            x_h = torch.cat((xTmp, Ht),0).cuda()
                
            Rt = torch.sigmoid(self.Wr.mm(x_h) + self.Br)
            Zt = torch.sigmoid(self.Wz.mm(x_h) + self.Bz)
            Ht_ = torch.mul(Ht, Rt)
            x_h_ = torch.cat((xTmp, Ht_),0).cuda()
            H_ = torch.tanh(self.Wh.mm(x_h_) + self.Bh)
            
            Ht = torch.mul(Zt, Ht)+torch.mul(1 - Zt, H_)
            y = self.W.mm(Ht) + self.b
            # no softmax: included in cross entropy loss
            y = y.transpose(1,0).contiguous()
            # y:  batch_size, vocab
            output.append(y)
            
        output = torch.cat(output,0)
        output = output.view(seq_len * batch_size, -1)
        #output:    (seq * batchsize, vocab)
        return output, Ht
          
    
def trainGRU():
    
    trainSet, devSet, vocab = preprocess()
    
    dataLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
    
    model = GRUPoem(len(vocab))
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
            
            output, _ = model(input_)
            #output:    (seq * batchsize, vocab)
            
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            lossSum += loss.cpu().detach().numpy()

            count += 1
            print("Epoch: %d batch index:%d loss: %.6f perplexity:%0.8f" % (epoch, idx, loss, torch.mean(torch.exp(loss)).item()))
        lossSet.append(lossSum/count)
        print("Perplexity after epoch %d : %.6f" % (epoch, pplxty))
#         if pplxty >= lastPplxty:
#             break
        
    return model, lossSet, pplxSet


model2 = trainGRU()

def drawPic(y):
    plt.figure()
    y = y.reshape(-1)
    x = np.arange(1,len(y)+1,1)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

drawPic(np.array(model2[1]))