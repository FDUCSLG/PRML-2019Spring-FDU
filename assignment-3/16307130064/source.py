import argparse
import pickle
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchnet import meter
import model
import utils
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, dest="data", help=".pkl file to use")
parser.add_argument("--name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="name", help="Name for this task")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")  
parser.add_argument("--embed_dim", default=128, dest="embed_dim", type=int, help="embed_dim")
parser.add_argument("--layers", default=1, dest="layers", type=int, help="layers")
parser.add_argument("--hidden_size", default=256, dest="hidden_size", type=int, help="hidden_size")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="dropout")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--batch-size", default=128, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--num_epochs", default=100, dest="num_epochs", type=int,
                    help="Number of full passes through training set")
                    
parser.add_argument("--lr", default=1e-3, dest="lr", type=float,help="learning rate")
parser.add_argument("--gpu", default=True, dest="gpu", action="store_false", help="gpu")
parser.add_argument("--API", default=False, dest="API", action="store_true", help="use my own lstm")
parser.add_argument("--test", dest="test", action="store_true", help="test")
options = parser.parse_args()
print(options)
data=pickle.load(open(options.data, "rb"))
w2i=data["w2i"]
i2w=utils.to_id_list(w2i)
max_length=data["max_length"]
train_data=data["train_data"]
dev_data=data["dev_data"]
print(len(train_data),len(dev_data),len(w2i),max_length)

if options.word_embeddings is not None:
    freeze=False
    print("freeze embedding:",freeze)
    init_embedding=utils.read_pretrained_embeddings(options.word_embeddings, w2i,options.embed_dim)
    embed=nn.Embedding.from_pretrained(torch.FloatTensor(init_embedding),freeze=freeze)
else: 
    embed=nn.Embedding(len(w2i),options.embed_dim)
    
model=model.Generator(embed,options.embed_dim,len(w2i),options.hidden_size,num_layers=options.layers, dropout=options.dropout,use_API=options.API)
optimizer=torch.optim.Adam(model.parameters(), lr = options.lr)
#optimizer=torch.optim.SGD(model.parameters(), lr = options.lr,momentum=0)
criterion = nn.CrossEntropyLoss()

if options.gpu:
    model=model.cuda()
    criterion=criterion.cuda()    

loss_meter = meter.AverageValueMeter()

if options.old_model:
    # incremental training
    print("Incremental training from old model: {}".format(options.old_model))
    model.load_state_dict(torch.load(options.old_model))
    
best_model="{}_{}_{}_{}".format(options.name, options.API, options.hidden_size, options.embed_dim)   
print(best_model)
if options.num_epochs>0:      
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = options.batch_size, shuffle = True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size = options.batch_size, shuffle = False)
    best=1e9
    count=0
    print("start training")
    for epoch in range(options.num_epochs):
        loss_meter.reset()
        model.train()
        #print(epoch)
        for _, data_ in enumerate(train_loader):    
            data_ = data_.long().contiguous()
            if options.gpu:
                data_ = data_.cuda()
            optimizer.zero_grad()
            input_,target = Variable(data_[:,:-1]),Variable(data_[:,1:])
            output, _  = model(input_)
            loss = criterion(output,target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

        tmp_loss = float(loss_meter.value()[0])
        
        model.eval()
        loss_meter.reset()
        for _, data_ in enumerate(dev_loader):    
            data_ = data_.long().contiguous()
            if options.gpu:
                data_ = data_.cuda()
            input_,target = Variable(data_[:,:-1]),Variable(data_[:,1:])
            output, _  = model(input_)
            loss = criterion(output,target.contiguous().view(-1))
            loss_meter.add(loss.item()) 
            
        ppl = math.exp(loss_meter.value()[0])
        print('Epoch {:d} | loss{:5.2f} | ppl {:8.2f}'.format(epoch, tmp_loss, ppl))
        if ppl<best:
            best=ppl
            count=0
            torch.save(model.state_dict(),best_model)
        else:
            count+=1
            
        if count>10:
            print("ppl didn't increase within 20 epochs")
            break
            
    model.load_state_dict(torch.load(best_model))
    
def generate(start_words,prefix_words=None,temp=0.6):
    model.eval()
    results = list(start_words)
    start_word_len = len(start_words)
    inp = torch.Tensor([w2i['<sos>']]).view(1,1).long()
    if options.gpu:
        inp=inp.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output,hidden = model(inp,hidden)
            inp = Variable(inp.data.new([w2i[word]])).view(1,1)
    
    for i in range(400):
        output,hidden = model(inp,hidden)
  
        if i < start_word_len:
            w = results[i]
            inp = Variable(inp.data.new([w2i[w]])).view(1,1)      
        else:
            #top_index  = output.data[0].topk(1)[1][0]
            prob = torch.exp(output[0]/temp)
            top_index = prob.multinomial(1)
            w = i2w[top_index]  
            results.append(w)
            inp = Variable(inp.data.new([top_index])).view(1,1)
        if w=='<eos>':
            del results[-1]
            break     
    return ''.join(results)

starts=["日","红","山","夜","湖","海","月"]
for word in starts:
    print(generate(word))
    
