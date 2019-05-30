import torch
import random
from config import *


def generate(model, start_word, vocab):
    result = []
    input_ = torch.Tensor([vocab.to_index(start_word)]).view(1, 1).long()
    input_ = input_.cuda()
    hidden = None
    result.append(start_word)

    for i in range(Config.max_gen_len-1):
        output, hidden = model(input_, hidden)
        top_index = output.data[0].topk(5)
        index = random.randint(0, 4)
        wordix = top_index[1][index].item()
        while vocab.to_word(wordix) == ' ':
            index = random.randint(0, 4)
            wordix = top_index[1][index].item()
        w = vocab.to_word(wordix)
        result.append(w)
        input_ = input_.data.new([wordix]).view(1, 1)
    return result
