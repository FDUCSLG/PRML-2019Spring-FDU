import os
import json
from fastNLP import Vocabulary
import numpy as np
import re
from langconv import *
from torch.utils.data import DataLoader
import torch

def prepareData(path='./json'):
    files = os.listdir(path)
    poemList = []
    for file in files:
        if not os.path.isdir(file):
            if "tang" in file and "authors" not in file:
                tmpPath = path +"/"+file
                with open(tmpPath, encoding='utf-8') as doc:
                    poemSet = json.load(doc)
                    for poemInf in poemSet:
                        poemPara = poemInf['paragraphs']  
                        poem = ""
                        for sentence in poemPara:
                            #delete all invalid words
                            rule = re.compile(u'[0-9a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|:：]')
                            sentence = re.sub(rule, "", sentence)
                            poem += sentence
                        poemList.append(poem)
    return poemList

def traditional2simplified(poemList):
    for i in range(len(poemList)):
        poemList[i] = Converter('zh-hans').convert(poemList[i])
    return poemList
    
def prepareVocab(poemList):
    vocab = Vocabulary()
    for poem in poemList:
        for character in poem:
            vocab.add(character)
    vocab.build_vocab()
    return vocab
            
    
def word2idx(poemList, vocab, maxLen=80):
    train_data = []
    for poem in poemList:
        if len(poem) >= maxLen:
            poem = poem[:maxLen]
        else:
            poem += ' '*(maxLen - len(poem))
        numList = []
        for character in poem:
            charIdx = vocab.to_index(character)
            numList.append(charIdx)
        train_data.append(numList)
    output = np.array(train_data)
    return output