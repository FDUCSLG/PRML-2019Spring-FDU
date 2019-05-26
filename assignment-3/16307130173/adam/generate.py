import torch
import random
from lstm_utils import *


def generate(model, start_word, word_dict, sentence_length, single_length):
    res_lst = []
    X = torch.Tensor([word_dict.to_index(start_word)]).view(1, 1).long()
    X = X.cuda()
    hidden = None
    res_lst.append(start_word)

    cnt = 1
    cc = 0
    print('Start Generating')

    for i in range(sentence_length):

        output, hidden = model(X, hidden)
        
        top_k = output.data[0].topk(10)

        index = random.randint(0, 9)
        new_word = top_k[1][index].item()
        
        while word_dict.to_word(new_word) == '~':
            index = random.randint(0, 9)
            new_word = top_k[1][index].item()

        cnt += 1
        res_lst.append(word_dict.to_word(new_word))
        if cnt % ((sentence_length + 1) // single_length) == 0:
            cc += 1
            if cc % 2 == 1:
                res_lst.append('， ')
            else:
                res_lst.append('。 ')
        
        X = X.data.new([new_word]).view(1, 1)
    
    return res_lst
