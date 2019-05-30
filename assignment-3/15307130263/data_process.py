import fastNLP
import torch
import re
from fastNLP import DataSet
from fastNLP import Vocabulary
from fastNLP import Instance
import numpy as np

def cleantxt(raw):
	fil = re.compile(u'[^\u4e00-\u9fff:]+', re.UNICODE)
	return fil.sub('', raw)

class TangPoemDataset:
    def __init__(self,useBigData = True,useSmallData = False,maxLength = None):
        fileName1 = '../handout/tangshi.txt'

        self.datas = []
        self.totalWords = 0
        if useSmallData:
            fd = open(fileName1,'r',encoding='UTF-8')
            lines = fd.readlines()
            for line in lines:
                line = line.strip()
                line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",line)
                line = cleantxt(line)
                if line == '':
                    continue
                self.datas.append([])
                for char in line:
                    self.datas[-1].append(char)
                #self.datas[-1].append("<EOS>")
            fd.close()

        if useBigData:
            fileName2 = './tangshi43030.txt' # 全唐诗补充数据
            fd = open(fileName2,'r',encoding='UTF-8')
            lines = fd.readlines()
            for line in lines:
                line = line.strip()
                line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",line)
                line = cleantxt(line)
                line = line.split(':')[-1]
                if line == '':
                    continue
                self.datas.append([])
                for char in line:
                    self.datas[-1].append(char)
                #self.datas[-1].append("<EOS>")
            fd.close();

        if maxLength:
            self.dealMaxLength(maxLength)
        print('read datas')

        # build vocab
        self.vocab = Vocabulary(max_size=7561,min_freq=3,unknown='<unk>',padding=None)
        for line in self.datas:
            for char in line:
                self.vocab.add(char)
        self.vocab.build_vocab()
        self.totalWords = len(self.vocab)
        print(self.totalWords)
        self.weight = [0 for i in range(self.totalWords)]
        for line in self.datas:
            for i in range(len(line)):
                line[i] = self.vocab.to_index(line[i])
                self.weight[line[i]] += 1
        
        minFreq = self.weight[-1]
        self.weight = [float(minFreq) / item for item in self.weight]
        self.weight = torch.Tensor(self.weight)
        print('built vocab')

        self.dataset = DataSet()
        for item in self.datas:
            self.dataset.append(Instance(input_s=item[:-1],output_s=item[1:]))
        self.dataset.set_input('input_s','output_s')
        self.dataset.set_target('output_s')
        self.trainSet,self.testSet = self.dataset.split(0.2)
        print('built dataset')
    
    def loadCharEmbedding(self,fileName='./sikuquanshuword.txt'):
        fd = open(fileName,'r',encoding='UTF-8')
        fline = fd.readline()
        flist = fline.strip().split(' ')
        total = int(flist[0])
        self.dim = int(flist[1])
        self.embedding = np.random.random((self.totalWords,self.dim))
        unkid = self.vocab.to_index('<unk>')
        count = 0
        for i in range(total):
            line = fd.readline()
            flist = line.strip().split(' ')
            word = flist[0]
            idx = self.vocab.to_index(word)
            if idx == unkid:
                continue
            else:
                count += 1
                for i in range(self.dim):
                    self.embedding[idx][i] = float(flist[i + 1])
        print(count,'words embedded with total ',self.totalWords,'words')

    def dealMaxLength(self,length):
        ret = []
        for line in self.datas:
            ret = ret + self.dealAString(line,length)
        self.datas = ret
    
    def dealAString(self,lists,length,step = 10):
        ret = []
        for s in range(0,len(lists) - length + 1,step):
            ret.append(lists[s:s+length] + ['<EOS>'])
        return ret
        if len(lists) < length:
            return []
        elif len(lists) == length:
            ret = lists
            ret.append('<EOS>')
            return [ret]
        else:
            tmp = lists[:length]
            tmp.append('<EOS>')
            ret.append(tmp)

        hret = self.dealAString(lists[length:],length)
        ret = ret + hret
        return ret