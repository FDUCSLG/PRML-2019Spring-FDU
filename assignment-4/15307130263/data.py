import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import os
os.sys.path.append('..')
from handout import *
import torch
import json

def getCharDataset():
    d_train,d_test = get_text_classification_datasets()
    vocab = buildVocab(d_train,d_test)
    ret0 = dealDatas(d_train,vocab)
    ret1 = dealDatas(d_test,vocab)
    return ret0,ret1

def buildVocab(data0,data1):
    vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    vocab = {}
    for i in range(len(vocabulary)):
        vocab[vocabulary[i]] = i
    return vocab

def dealDatas(datas,vocab):
    retDataset = DataSet()
    length = len(datas.target)
    for i in range(length):
        charData = datas.data[i]
        cidxList = []
        for char in charData:
            if char in vocab:
                cidxList.append(vocab[char])
            else:
                cidxList.append(68)

        if len(cidxList) > 1014:
            cidxList = cidxList[:1014]
        else:
            while len(cidxList) < 1014:
                cidxList.append(68)
        retDataset.append(Instance(char_data=cidxList,label=int(datas.target[i])))
    retDataset.set_input('char_data','label')
    retDataset.set_target('label')
    return retDataset

def getWordDataset():
    vocab,embedding = loadEmbedding('./glove.6B/glove.6B.200d.txt')
    d_train,d_test = get_text_classification_datasets()
    return dealWordDatas(d_train,vocab),dealWordDatas(d_test,vocab),embedding

def loadEmbedding(filename):
    fd = open(filename,'r',encoding='UTF-8')
    lines = fd.readlines()
    vocab = {}
    embedding = []
    count = 0
    for line in lines:
        nlist = line.strip().split(' ')
        vocab[nlist[0]] = count
        count += 1
        for i in range(1,len(nlist)):
            nlist[i] = float(nlist[i])
        embedding.append(nlist[1:])
    embedding = np.array(embedding)
    embedding = np.append(embedding,np.zeros((1,200)),0)
    print(embedding.shape)
    return vocab,embedding

def dealWordDatas(datas,vocab,maxlength = 128):
    retDataset = DataSet()
    length = len(datas.target)
    for i in range(length):
        words = datas.data[i].strip().split(' ')
        widxList = []
        count = 0
        for word in words:
            if word in vocab:
                widxList.append(vocab[word])
                count += 1
            if count == maxlength:
                break
        while count < maxlength:
            widxList.append(400000)
            count += 1
        retDataset.append(Instance(word_data=widxList,label=int(datas.target[i])))
    retDataset.set_input('word_data','label')
    retDataset.set_target('label')
    return retDataset
        

if __name__ == '__main__':
    datas0,datas1 = getWordDataset()