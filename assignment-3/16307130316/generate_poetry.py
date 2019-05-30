import torch
import pickle as p
import torch.autograd as autograd
import numpy


def invert_dict(d):
    return dict((v, k) for k, v in d.items())


def make_index(word, word_to_ix):
    rst = autograd.Variable(torch.LongTensor([word_to_ix[word]]).view(1, -1))
    return rst


def generate(startWord='<START>', max_length=95):
    model = torch.load('./model_padding_10k/poetry-gen-epoch38-loss4.066066.pt')
    with open('wordDic', 'rb') as f:
        word_to_ix = p.load(f)
    ix_to_word = invert_dict(word_to_ix)
    input = make_index(startWord, word_to_ix)
    poem = ""
    if startWord != "<START>":
        poem = startWord
    hidden = None
    for i in range(max_length):
        output, hidden = model(input, hidden)
        topvs, topis = output[0].data.topk(2)
        topi = topis[0][0].item()
        topi1 = topis[0][1].item()
        w = ix_to_word[topi]
        w1 = ix_to_word[topi1]
        if w == "<EOS>":
            # w = w1
            w = ix_to_word[numpy.random.randint(low=2, high=len(ix_to_word)-4)]
            hidden = None
            # w = output_name[-2]
        if w != "<EOS>":
            poem += w
        input = make_index(w, word_to_ix)
    return poem
