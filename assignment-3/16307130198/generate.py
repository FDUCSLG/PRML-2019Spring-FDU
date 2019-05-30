import _pickle as pickle
import config
import sys
import os
import torch
import models
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import tqdm
import utils

import torch.nn.functional as F
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase

from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler 
from fastNLP import Trainer
from copy import deepcopy
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Adam
from fastNLP.core.callback import EarlyStopCallback 



def generate_way1(model, start_words, ix2word, word2ix, prefix_words=None, temperature=0.8):
    results = ""
    start_word_len = len(start_words)
    start_input = Variable(torch.Tensor([word2ix['<start>']]).view(1,1).long())
    hidden = None

    index = 0
    pre_word = '<start>'
    
    start_input = start_input
    #start_input = start_input.cuda()
    max_length = 200 
    for i in range(max_length):
        ret_dict = model(start_input, hidden)
        
        output = ret_dict["output"] 
        hidden = ret_dict["hidden"]
        

        top_index = output.data[0].topk(1)[1][0]


        top_w = ix2word[int(top_index.cpu().numpy())]
        
        output = output.data[0].div(temperature).exp()
        _index = torch.multinomial(output, 1)[0].cpu().numpy()
        w = ix2word[int(_index)]
        if top_w in ['。','，','？','！']:
            w = top_w

        if(pre_word in ['<start>', '。']):
            if index == start_word_len:
                break
            else:
                #print(index)
                w = start_words[index]
                index += 1
                start_input = Variable(start_input.data.new([word2ix[w]])).view(1,1)
        else:
            start_input = Variable(start_input.data.new([word2ix[w]])).view(1,1)
        
        results+=w
        pre_word = w
    return results 

def generate_way2(model, start_words, ix2word, word2ix, prefix_words=None, 
                  temperature=0.8):
    results = ""
    start_word_len = len(start_words)
    start_input = Variable(torch.Tensor([word2ix['<start>']]).view(1,1).long())
    hidden = None

    index = 0
    pre_word = '<start>'
    
    start_input = start_input
    #start_input = start_input.cuda()
    max_length = 200
    max_sentence_num = 8
    sentence_num = 0
    for i in range(max_length):
        ret_dict = model(start_input, hidden)
        
        output = ret_dict["output"] 
        hidden = ret_dict["hidden"]
        
        top_index = output.data[0].topk(1)[1][0]
        top_w = ix2word[int(top_index.cpu().numpy())]
        
        output = output.data[0].div(temperature).exp()
        _index = torch.multinomial(output, 1)[0].cpu().numpy()
        w = ix2word[int(_index)]

        if top_w in ['。','，','？','！']:
            w = top_w

        if(pre_word in ['<start>']):
            w = start_words[0]
            start_input = Variable(start_input.data.new([word2ix[w]])).view(1,1)
        elif w in ['<end>']:
             return results
        else: 
            if w in ['。','，','？','！']:
                sentence_num += 1

            start_input = Variable(start_input.data.new([word2ix[w]])).view(1,1)
        
        results+=w
        if sentence_num >= max_sentence_num:
            return results
        pre_word = w
    return results 



if __name__ == '__main__':
    opt = config.Config()
    vocab = pickle.load(open(opt.vocab, 'rb'))
    word2idx = vocab.word2idx
    idx2word = vocab.idx2word
    vocab_size = len(word2idx)
    #generate_words = ['杂','乱','无','章']
    generate_words = ['日' ,'红','山','夜','湖','海','月']
    #generate_words = ['文','娇','困','了']
    embedding_dim = 512
    hidden_dim = 512
    model_path = "./model_log/best_PoetryModel_pp_2019-05-25-19-55-26"
    model = utils.find_class_by_name(opt.model_name, [models])(
            vocab_size, embedding_dim, hidden_dim
    )
    utils.load_model(model, model_path)
    
    #result = generate_way1(model, generate_words, idx2word, word2idx, None, temperature=0.8)
    #print(result)
    
    result_list = []
    
    for i in generate_words:
        result_list.append(generate_way2(model, i, idx2word, word2idx, None, temperature=0.6))
    
    for poetry in result_list:
        print(poetry) 
        print("\n")







