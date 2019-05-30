import numpy as np
import pickle
import os
import json
import random

w2i={}
w2i["<sos>"]=0
w2i["<eos>"]=1
w2i["<unk>"]=2
w2i["<pad>"]=3
max_len=80

def word_to_id(w):
    if w not in w2i:
        w2i[w]=len(w2i)
    return w2i[w]
    
what="tang"
res=[]
ed=57000
for i in range(0,ed,1000):
    filename="poet.{1}.{0}.json".format(i,what)
    with open(os.path.join("poetry",filename),"r",encoding="utf-8") as f:
        poems=json.load(f)
        for poem in poems:
            sents=list("".join(poem["paragraphs"]))
            #print(sents)
            sents=["<sos>"]+sents[:max_len]+["<eos>"]
            off=max_len+2-len(sents)
            if off>0:
                sents.extend(["<pad>"]*off)
            sents=[word_to_id(w) for w in sents]
            res.append(sents)
            
random.shuffle(res)
n=int(len(res)*0.8)
train,dev=res[:n],res[n:]
print(len(w2i),len(train),len(dev))
dic={}
dic["w2i"]=w2i
dic["train_data"]=np.array(train)
dic["dev_data"]=np.array(dev)
dic["max_length"]=max_len+2
with open("{}_{}.pkl".format(what,max_len),"wb") as outfile:
    pickle.dump(dic,outfile)
           
