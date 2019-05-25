import sys
sys.path.append("../")
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import torch.utils.data as data
import torch
import numpy as np
import pickle
import config

def construct_dataset(sentences):
    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        dataset.append(instance)
    return dataset

def cut_pad(ins):
    x = ["<START>"] + list(ins['raw_sentence']) + ["<EOS>"]
    x = x[0:config.MAXLEN]
    length = len(x)
    if length < config.MAXLEN:
        pad = (config.MAXLEN - length) * ["<pad>"]
        x = pad + x
    return x

def Get_Data_Vocab(path):
    s = ""
    with open (path, "r", encoding='UTF-8') as f:
        for line in f:
            s += line.rstrip('\r\n') + "#"

    sentences = s.split("#")

    dataset = construct_dataset(sentences)
    dataset.apply(cut_pad, new_field_name='words') #控制每首诗长度一致
    # 分出测试集、训练集
    dev_data, train_data = dataset.split(0.8)
    # 构建词表, Vocabulary.add(word)
    vocab = Vocabulary(padding="<pad>", min_freq=2)
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
    vocab.build_vocab()
    print(vocab.idx2word)

    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
    train_data.apply(lambda x: x['words'][:-1], new_field_name="input")
    train_data.apply(lambda x: x['words'][1:], new_field_name="target")
    dev_data.apply(lambda x: x['words'][:-1], new_field_name="input")
    dev_data.apply(lambda x: x['words'][1:], new_field_name="target")
    train_data.set_input("input")
    train_data.set_target("target")
    dev_data.set_input("input")
    dev_data.set_target("target")

    return vocab, train_data, dev_data

class Poems_Set(data.Dataset):
    def __init__(self, data):
        self.data_ = data
        
    def __getitem__(self, index):
        return np.array(self.data_[index]['words'])

    def __len__(self):
        return len(self.data_)

def Init_Dataloader(path=config.data_path, batch_size = None):
    vocab, train_data, dev_data = Get_Data_Vocab(path)
    train_data = Poems_Set(train_data)
    dev_data = Poems_Set(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size = batch_size,
                                             shuffle = True, 
                                             num_workers=2)
    develop_loader = torch.utils.data.DataLoader(dev_data,
                                             batch_size = batch_size,
                                             shuffle = True, 
                                             num_workers=2)
    return vocab, train_loader, develop_loader

if __name__ == "__main__":
    vocab, train_data, dev_data = Get_Data_Vocab(config.data_path)
    pickle.dump(vocab, open(config.vocab_path, "wb"))  
    pickle.dump(train_data, open(config.train_data_path, "wb"))
    pickle.dump(dev_data, open(config.dev_data_path, "wb"))
