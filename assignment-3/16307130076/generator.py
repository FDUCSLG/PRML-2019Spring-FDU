import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from prepare_data import *

def generate(model, start_word, idx2word, word2idx, gen_len,T,k):
    if start_word is None:
        input = torch.randint(len(idx2word), size=(1, 1))
    else:
        start_word_id = w2i(word2idx, start_word)
        if start_word_id == word2idx['<unk>']:
            raise Exception("Start word is not in the vocab.")
            return
        else:
            input = torch.Tensor([start_word_id]).view(1,1).long()
    net=model
    use_cuda = torch.cuda.is_available()
    results=[start_word]
    if use_cuda:
        net.cuda()
        input = input.cuda()
    net.eval()
    for i in range(1,gen_len):
        outputs=net(input)
        outputs = F.softmax(outputs / T, dim=1)
        outputs = torch.topk(outputs, dim=1,k=k)[1]
        output = outputs[-1][random.randint(0,k-1)].item()
        if idx2word[output] == '<unk>' or idx2word[output] == '\n':
            break
        else:
            results.append(idx2word[output])
            a_i = torch.Tensor([output]).view(1, 1).long()
            if use_cuda:
                a_i=a_i.cuda()
            input = torch.cat((input,a_i),dim=1)
    return results

            


