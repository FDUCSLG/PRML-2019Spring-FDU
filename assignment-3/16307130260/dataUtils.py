import numpy as np
from fastNLP import Vocabulary

def get_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    data, _, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
    wordlist = []
    for d in data:
        for ix in d:
            wordlist.append(ix2word[ix])
    vocab = Vocabulary(min_freq=10, padding="</s>")
    vocab.add_word_lst(wordlist)
    vocab.build_vocab()
    # vocab = Vocabulary(min_freq=10, padding="</s>").add_word_lst(wordlist).build_vocab()
    vocab_size = len(vocab.word2idx)
    for d in data:
        for i in range(len(d)):
            # d[i] = vocab[vocab.to_word(d[i])]
            if d[i] >= vocab_size:
                d[i] = vocab["<unk>"]
    
    print(vocab_size)

    return data, vocab