import numpy as np
import os

def processing(file_name='dataset/tangshi.txt'):
    file = open(file_name, 'r')
    line = file.readline()
    poem = ''
    while line:
        line = line.replace('，', '').replace('。', '')
        poem += line
        line = file.readline()

    tmp_poem = poem[0]
    for i in range(1, len(poem)):
        if poem[i-1] == '\n' and poem[i] == '\n':
            tmp_poem += '!'
        else:
            tmp_poem += poem[i]

    poem_list = tmp_poem.replace('\n', '').split('!')
    return poem_list

def get_dict(data):
    words = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            words.append(data[i][j])
    words = set(words)
    word2idx = {word: idx for (idx, word) in enumerate(words)}
    word2idx['<EOS>'] = len(word2idx)
    word2idx['<OOV>'] = len(word2idx)
    word2idx['<START>'] = len(word2idx)
    idx2word = {idx: word for (word, idx) in list(word2idx.items())}
    return word2idx, idx2word

def pad_sentence(sentence, seq_len):
    if len(sentence) > seq_len:
        return sentence[:seq_len]
    pad_len = seq_len - len(sentence) 
    for i in range(pad_len):
        sentence = ['<OOV>'] + sentence
    return sentence

def padding(data, word2idx, idx2word, seq_len=128):
    data = [list(dat) for dat in data]
    data = [['<START>'] + dat + ['<EOS>'] for dat in data]
    data = [pad_sentence(dat, seq_len) for dat in data]
    data_idx = np.zeros([len(data), seq_len])
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_idx[i][j] = word2idx[data[i][j]]
    return data_idx.astype(np.int32)

def get_data_mini(path='dataset/tang_mini.npz', file_name='dataset/tangshi.txt'):
    if os.path.exists(path):
        data = np.load(path)
        data, word2idx, idx2word = data['data'], data['word2idx'].item(), data['idx2word'].item()
        #print(':-)')
        return data, word2idx, idx2word
    
    data = processing(file_name)
    word2idx, idx2word = get_dict(data)
    data = padding(data, word2idx, idx2word)
    print(type(data[0][0]))
    np.savez_compressed(path,
                        data=data,
                        word2idx=word2idx,
                        idx2word=idx2word)
    return data, word2idx, idx2word

def get_data_big(path='dataset/tang_big.npz', file_name='dataset/tangshi.txt'):
    if os.path.exists(path):
        data = np.load(path)
        data, word2idx, idx2word = data['data'], data['word2idx'].item(), data['idx2word'].item()
        #print(':-)')
        return data, word2idx, idx2word
    else:
        print(':-(')
        return get_data_mini(path, file_name)