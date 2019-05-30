import os
import re
import numpy as np
from fastNLP import Vocabulary


def pre_process():
    path = os.getcwd()+'/poem.txt'

    with open(path, encoding='utf-8') as file:
        sentence = file.readlines()
        data = list(sentence)

    poem = []
    line = 0
    for unit in data:
        line += 1
        if line % 2 == 0:
            tmp = re.sub('，', '', unit)
            sent = re.sub('。', '', tmp)
            poem.append(sent[:-1])

    padding_poem = []
    for sentence in poem:
        if len(sentence) < 80:
            sentence = ' '*(80-len(sentence)) + sentence
            padding_poem.append(sentence)
        else:
            padding_poem.append(sentence[:80])

    vocab = Vocabulary()
    for line in padding_poem:
        for character in line:
            vocab.add(character)
    vocab.build_vocab()

    train_data = []
    for poetry in padding_poem:
        p = []
        for char in poetry:
            p.append(vocab.to_index(char))
        train_data.append(p)
    train_data = np.array(train_data)
    return vocab, train_data

