# coding: utf-8
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.io import dataset_loader
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler
from get_data import *
import numpy as np
import string
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from matplotlib import pyplot as plt

SEED = 233
torch.manual_seed(SEED)

def remove_empty(x):
    while '' in x['text']:
        x['text'].remove('')
    return x['text']

def pad_label(x, n_cat=4):
    label = [0 for _ in range(n_cat)]
    label[x['label']] = 1
    return label

class Engine(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_iterator, self.valid_iterator, self.test_iterator =\
            self.init_data_iterator()
        
        self.model, self.optimizer = self.init_model_optimizer()

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = self.criterion.to(self.device)

    def init_data_iterator(self, prop=0.8):
        train_data, test_data = get_text_classification_datasets()
        train_dataset = DataSet()
        valid_dataset = DataSet()
        length = len(train_data.data)
        for i in range(length):
            if i < int(prop*length):
                train_dataset.append(Instance(text=train_data.data[i], label=int(train_data.target[i])))
            else:
                valid_dataset.append(Instance(text=train_data.data[i], label=int(train_data.target[i])))

        test_dataset = DataSet()
        for i in range(len(test_data.data)):
            test_dataset.append(Instance(text=test_data.data[i], label=int(test_data.target[i])))

        trans = str.maketrans({key: None for key in string.punctuation})

        train_dataset.apply(lambda x: x['text'].lower().translate(trans), new_field_name='text')
        train_dataset.apply(lambda x: re.sub(pattern=r'\s', repl=' ', string=x['text']), new_field_name='text')
        train_dataset.apply(lambda x: x['text'].split(' '), new_field_name='text')
        train_dataset.apply(remove_empty, new_field_name='text')
        train_dataset.apply(pad_label, new_field_name='label_pad')

        valid_dataset.apply(lambda x: x['text'].lower().translate(trans), new_field_name='text')
        valid_dataset.apply(lambda x: re.sub(pattern=r'\s', repl=' ', string=x['text']), new_field_name='text')
        valid_dataset.apply(lambda x: x['text'].split(' '), new_field_name='text')
        valid_dataset.apply(remove_empty, new_field_name='text')
        valid_dataset.apply(pad_label, new_field_name='label_pad')

        test_dataset.apply(lambda x: x['text'].lower().translate(trans), new_field_name='text')
        test_dataset.apply(lambda x: re.sub(pattern=r'\s', repl=' ', string=x['text']), new_field_name='text')
        test_dataset.apply(lambda x: x['text'].split(' '), new_field_name='text')
        test_dataset.apply(remove_empty, new_field_name='text')
        test_dataset.apply(pad_label, new_field_name='label_pad')

        vocab = Vocabulary(min_freq=10)
        train_dataset.apply(lambda x: [vocab.add(word) for word in x['text']])
        vocab.build_vocab()

        train_dataset.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name='text_index')
        valid_dataset.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name='text_index')
        test_dataset.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name='text_index')

        train_dataset.set_input('text_index')
        train_dataset.set_target('label_pad')

        valid_dataset.set_input('text_index')
        valid_dataset.set_target('label_pad')

        test_dataset.set_input('text_index')
        test_dataset.set_target('label_pad')

        bs = self.args['data']['batch_size']
        train_batch = Batch(dataset=train_dataset, batch_size=bs, sampler=RandomSampler())
        valid_batch = Batch(dataset=valid_dataset, batch_size=bs, sampler=RandomSampler())
        test_batch = Batch(dataset=test_dataset, batch_size=bs, sampler=RandomSampler())

        self.input_dim = len(vocab)

        return train_batch, valid_batch, test_batch

    def init_model_optimizer(self):
        model = models.__dict__[self.args['model']['arch']](
            self.input_dim,
            self.args['model']['params'])
        model = model.to(self.device)

        optimizer = optim.__dict__[self.args['model']['optimizer']](
            params=model.parameters(),
            lr=self.args['model']['learning_rate'],
            weight_decay=self.args['model']['weight_decay'])
        return model, optimizer

    def load_model(self):
        ckpt = torch.load(self.args['model']['path']+self.args['name']+'.ckpt')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print('Model loaded.')
        
    def save_model(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.args['model']['path']+self.args['name']+'.ckpt')
        print('Model saved.')

    def load_pretrained_model(self):
        print('Loading pre-trained model ...')
        pre_train = gensim.models.KeyedVectors.load_word2vec_format(self.args['data']['pretrain_path'])
        weights = torch.FloatTensor(pre_train.vectors)
        self.model.embedding.weight = torch.nn.Parameter(weights)
        print('Finish loading.')

    def accuracy(self, preds, y):
        max_preds = torch.argmax(preds, dim=-1)
        max_y = torch.argmax(y, dim=-1)
        correct = (max_preds == max_y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train_step(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        i = 0
        for X, y in iterator:
            #print(i)
            i += 1
            label = y['label_pad'].float()
            self.optimizer.zero_grad()
            predictions = self.model(X['text_index'].transpose(0, 1)).squeeze(1)
            loss = self.criterion(predictions, label)
            acc = self.accuracy(predictions, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_step(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in iterator:
                label = y['label_pad'].float()
                predictions = self.model(X['text_index'].transpose(0, 1)).squeeze(1)
                loss = self.criterion(predictions, label)
                acc = self.accuracy(predictions, label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self):
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        print('Begin Training ...')
        for epoch in range(self.args['n_epoch']):
            train_loss, train_acc = self.train_step(self.train_iterator)
            valid_loss, valid_acc = self.evaluate_step(self.valid_iterator)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_acc.append(valid_acc)
            print('Epoch: ', epoch+1, \
                'Train Loss is:', train_loss, \
                'Train Acc is:', train_acc, \
                'Val Loss is:', valid_loss, \
                'Val Acc is: ', valid_acc
                )
        self.plot()

    def test(self):
        test_loss, test_acc = self.evaluate_step(self.test_iterator)
        self.test_loss = test_loss
        self.test_acc = test_acc
        print('Test Loss is:', test_loss, \
            'Test Acc is: ', test_acc)

    def plot(self):
        nums = len(self.train_loss)
        x = range(nums)
        plt.suptitle('loss and accuracy curves in training procedure', fontsize=16)
        plt.title('model: %s, optimizer: %s, epochs: %d, learning rate: %f' % 
                 (self.args['model']['arch'],
                  self.args['model']['optimizer'],
                  self.args['n_epoch'], 
                  self.args['model']['learning_rate']), fontsize=10)
        plt.plot(x, self.train_loss, label='train loss', color='#FFA500')
        plt.plot(x, self.train_acc, label='train acc', color='#00FFCC')
        plt.plot(x, self.valid_loss, label='valid loss', color='#CC9933')
        plt.plot(x, self.valid_acc, label='valid acc', color='#00CC99')
        plt.legend()
        plt.show()