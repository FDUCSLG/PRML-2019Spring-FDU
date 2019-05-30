import os
import torch
import numpy as np
from torch.autograd import Variable
from model import PoetryModel
import pickle
import config

def generate(model, vocab , start_words = None, prefix_setence = None, max_size =8,  temperature = 1):
    input_ = Variable(torch.Tensor(([vocab.to_index('<START>')]))).view(1, 1).long()
    hidden, ceil = None, None
    
    if prefix_setence:
        for word in prefix_setence:
            _, hidden, ceil = model(input_, hidden, ceil)
            input_ = Variable(input_.data.new([vocab.to_index(word)])).view(1,1)
    
    results = start_words
    start_word_len = len(start_words)
    size = 0
    for i in range(config.MAX_GEN_LEN):
        output, hidden, ceil = model(input_, hidden, ceil).values()

        if i < start_word_len:
            w = results[i]
            input_ = input_.data.new([vocab.to_index(w)]).view(1,1)
        else:
            output = output.data.view(-1).div(temperature).exp()
            for _ in range(10):
                index = torch.multinomial(output, 1)[0]
                w = vocab.to_word(int(index))
                if w!="<unk>" and w!="<START>" and w!="<pad>":
                    break
            results += w
            input_ = input_.data.new([index]).view(1,1)
        if w == '。' or w == '?':
            size += 1
        if w == '<EOS>' or size == max_size :
            break
    
    return [results]

def load_model(model, save_path, model_name):
    model_path = os.path.join(save_path, model_name)
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)


if __name__ == "__main__":
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    model =  PoetryModel(len(vocab), config.intput_size, config.hidden_size)
    load_model(model, 
              save_path=config.save_path, 
              model_name="best_PoetryModel_PPL_2019-05-21-21-25-10")
    start_words = ["日","红","山","夜","湖","海","月"]

    for word in start_words:
        results = generate( model,vocab, 
                            start_words=word, 
                            max_size=config.MAX_GEN_SIZE, 
                            temperature=config.temperature)
        print(results)