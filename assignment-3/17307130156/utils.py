import random
import json
import torch

from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

from vocabulary import Vocabulary
from collections import namedtuple

BEGIN = '<BOS>'
END = '<EOS>'
UNKNOWN = '<OOV>'

tag_list = [BEGIN, END, UNKNOWN]

def json2poems(path):
    # path: path of json file, string-like
    # return: string
    data = json.loads(open(path).read())
    poems = []
    for p in data:
        paragraphs = p['paragraphs'] # type: list
        poem = ''.join(paragraphs)
        poems.append(poem)
    return poems

def jsons2poems(paths):
    # paths: [path1(string), ...]
    # return: [string, ...]
    poems = []
    for p in paths:
        poems += json2poems(p) 
    return poems


def poem2data(poem, vocab):
    # poem type: string, shape: (number of chars)
    # return: (LongTensor(int, ...), LongTensor(int, ...) 

    # token = torch.LongTensor([vocab[BEGIN]] + [vocab[w] for w in poem] + [vocab[END]])
    token = torch.LongTensor([vocab[w] for w in poem] + [vocab[END]])
    input_data = token[:-1]
    target_data = token[1:]
    return input_data, target_data
        
def poems2data(poems, vocab):
    '''
    :poems: [str, ...]
    :return: [(seq_len1, seq_len1), ...]
    '''

    return [poem2data(p, vocab) for p in poems]


# Select only len between 40 and 50
def get_poems(path, min_len=45, max_len=50):
    # :path: str
    # :return: [str, ...]

    data = json.loads(open(path).read())
    poems = []
    for p in data:
        paragraphs = p['paragraphs'] # type: list
        poem = ''.join(paragraphs)
        if len(poem) > max_len or len(poem) < min_len : continue
        poems.append(poem)
   
    return poems

def update_vocab(vocab, poems):
    # :poems: [str, ...]
    
    for p in poems:
        vocab.update(tag_list)
        words = [c for c in p]
        vocab.update(words)

def get_vocab(paths):

    vocab = Vocabulary(min_freq=10)

    for path in paths:
        poems = get_poems(path)
        update_vocab(vocab, poems)

    vocab.build_vocab()

    return vocab
        
class DataProvider():

    # div is divide dataset into trainset and development set div=0.8 means 80% is trainset
    def __init__(self, files, batch_size=50, padding_value=0, div=0.8):
        self.files = files
        self.batch_size = batch_size
        self.padding_value = padding_value
        self.vocab = get_vocab(files)
        self.div = div
        # print (self.vocab)


    def padded_batches(self, shuffle=True):

        for path in self.files:
            print (f'Reading file: {path}')

            dataset = poems2data(get_poems(path), self.vocab)
            dataset = [(torch.LongTensor(inp), torch.LongTensor(t)) for inp, t in dataset]

            if shuffle: random.shuffle(dataset)

            # split into devset and trainset
            pivot = int(self.div * len(dataset))
            self.devset = dataset[pivot:]
            dataset = dataset[: pivot]
            print (f'trainset: {len(dataset)}, devset: {len(self.devset)}')

            batch_size = self.batch_size
            num_data = len(dataset)
            num_batch = num_data // batch_size
            
            for start in range(0, num_data, batch_size):

                end = min(num_data, start + batch_size)

                sorted_batch = sorted(dataset[start:end], key=lambda t: len(t[0]), reverse=True)
                # print (sorted_batch[0][0].shape[0]) 
                sorted_lengths = [len(i) for i, t in sorted_batch]

                input_batch, target_batch = [inp for inp, t in sorted_batch], [t for inp, t in sorted_batch]

                pib, ptb = pad_sequence(input_batch, padding_value=self.padding_value), pad_sequence(target_batch, padding_value=self.padding_value)

                # print ('padded input batch: ', pib.shape)
                # print ('padded target batch: ', ptb.shape)
                yield pib, ptb, sorted_lengths

    # Every file will generate a new devset
    def devset_batches(self):

        dataset = self.devset
        # FIXME 
        batch_size = 1
        # batch_size = self.batch_size
        num_data = len(dataset)
        num_batch = num_data // batch_size

        for start in range(0, num_data, batch_size):

            end = min(num_data, start + batch_size)
            sorted_batch = sorted(dataset[start:end], key=lambda t: len(t[0]), reverse=True)
            sorted_lengths = [len(i) for i, t in sorted_batch]

            input_batch, target_batch = [inp for inp, t in sorted_batch], [t for inp, t in sorted_batch]

            pib, ptb = pad_sequence(input_batch, padding_value=self.padding_value), pad_sequence(target_batch, padding_value=self.padding_value)

            yield pib, ptb, sorted_lengths

if __name__ == '__main__':
    # Usage
    files = ['raw_data/poet.tang.0.json']

    provider = DataProvider(files, batch_size=50, padding_value=0)


    for input_batch, target_batch, sorted_lengths in provider.padded_batches():
        print (input_batch.shape)
        print (target_batch.shape)
        print (input_batch)
        print (target_batch)
        input()


