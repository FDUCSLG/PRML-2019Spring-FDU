import random
import math
import torch
import torch.nn as nn

from torch.autograd import Variable

from train import model, vocab, provider


randn = 5
similarity = 0.1

def argmax(probs):
    li = [(i, prob) for i, prob in enumerate(probs)]
    li.sort(key=lambda x: x[1], reverse=True)
    for i, (idx, p) in enumerate(li):

        # This guarantee to produce punctuations , . ? at right position, which is with high p
        if math.exp(p) > 1 - similarity: return idx

        dis = [x for x in range(randn)]
        random.shuffle(dis)
        for di in dis:
            ridx = li[i + di][0]
            if ridx != vocab.unknown_idx:
                return li[i+di][0]
    
    return vocab.unknown_idx

def parse_batch(batch_data):
    # idx = torch.argmax(batch_data, dim=2).t().tolist()
    idx = [ [argmax(j) for j in i ] for i in batch_data.tolist()]
    seqs = [''.join([vocab.to_word(idx) for idx in batch]) for batch in idx]

    return seqs


def get_next(pre_words):

    model.zero_grad()
    state = model.init_hidden(1)
    # batch_size should be 1 since we generate each word in sequence
    sorted_lengths = [1]

    ouput_batch = None
    words = [word for word in pre_words]
    print (words)

    next_word = ''

    # input pre_words
    for w in words:
        yield w
        input_batch = torch.LongTensor([provider.vocab[w]]).view(1, 1)
        input_batch = Variable(input_batch)

        output_batch, state = model(input_batch, sorted_lengths)
        output_seqs = parse_batch(output_batch)
        next_word = output_seqs[0]
        

    yield next_word
    # predict
    while True:
        input_batch = torch.LongTensor([provider.vocab[next_word]]).view(1, 1)
        input_batch = Variable(input_batch)

        output_batch, state = model(input_batch, sorted_lengths) 
        output_seqs = parse_batch(output_batch)

        next_word = output_seqs[0]
        # [0] since our batch size is only 1 
        yield next_word


print ('Please input some words: ')
words = input()


output = ''
cnt = 0
for w in get_next(words):
    cnt += 1 
    output += w
    if w == '<EOS>' or cnt > 50: break
    if w == '。': output += '\n'

print (output)
    




