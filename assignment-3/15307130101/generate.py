import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from model import *
import tqdm
from config import *
from test import *


def generate(model, start_words, ix2word, word2ix, config):
    results = list(start_words)
    start_words_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    input.to(config.device)
    hidden = None
    for i in range(config.max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            return results[:-1]
    return results

