import re
import string
import json
import torch
import torch.utils.data
from fastNLP import DataSet, Instance, Vocabulary


def w2i(word2idx, word):
    if word2idx.__contains__(word):
        return word2idx[word]
    else:
        return word2idx['<unk>']


def import_data(path):
    dataset = DataSet()
    with open(path, encoding='utf-8') as f:
        content = f.readlines()[1::2]
        for c in content:
            c = re.sub(
                "[\s+\.\!\/_,$%^*(+\"\']+|[+——！？、~@#￥%……&*（）《》]+", "", c)  # ！，。？
            c = c.replace(string.whitespace, '').strip()
            dataset.append(Instance(raw_sentence=c))
    dataset.drop(lambda x: len(list(x['raw_sentence'])) == 0)

    def split_sent(ins):
        return list(ins['raw_sentence'])
    dataset.apply(split_sent, new_field_name='words', is_input=True)
    test_data, train_data = dataset.split(0.8)
    print(len(test_data), len(train_data))
    return test_data, train_data


def construct_vec(test_data, train_data):
    try:
        f = open('idx2word.json', 'r', encoding='utf-8')
        idx2word = json.load(f)
        idx2word = {int(k): v for k, v in idx2word.items()}
        f = open('word2idx.json', 'r', encoding='utf-8')
        word2idx = json.load(f)
        word2idx = {k: int(v) for k, v in word2idx.items()}
    except OSError:
        vocab = Vocabulary(min_freq=2, unknown='<unk>', padding='<pad>')
        train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()
        idx2word = vocab.idx2word
        word2idx = vocab.word2idx
        f = open('idx2word.json', 'w', encoding='utf-8')
        json.dump(idx2word, f)
        f = open('word2idx.json', 'w', encoding='utf-8')
        json.dump(word2idx, f)

    train_data.apply(lambda x: [
                     w2i(word2idx, word) for word in x['words']], new_field_name='word_seq', is_input=True)
    test_data.apply(lambda x: [
                    w2i(word2idx, word) for word in x['words']], new_field_name='word_seq', is_input=True)
    return test_data, train_data, idx2word, word2idx


class PoemSet(torch.utils.data.Dataset):
    def __init__(self, dataset, seq_len, seq_step, idx2word, word2idx):
        def reg_data(dataset, seq_len, seq_step, idx2word, word2idx):
            reg_dataset = None
            for idx, ins in enumerate(dataset):
                seq = ins['word_seq']
                print('processing %d/%d' % (idx, len(dataset)))
                if seq_len < len(seq):
                    for i in range(0, max(0, len(seq)-seq_len), seq_step):
                        t = torch.Tensor(seq[i:i+seq_len]).long()
                        t = t.unsqueeze(0)
                        if reg_dataset is None:
                            reg_dataset = t
                        else:
                            reg_dataset = torch.cat((reg_dataset, t), 0)
                else:
                    t = torch.Tensor(
                       [w2i(word2idx, '<pad>')]*(seq_len-len(seq))+seq).long()
                    t = t.unsqueeze(0)
                    if reg_dataset is None:
                        reg_dataset = t
                    else:
                        reg_dataset = torch.cat((reg_dataset, t), 0)
            return reg_dataset
        self.dataset = reg_data(dataset, seq_len, seq_step, idx2word, word2idx)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.size()[0]
