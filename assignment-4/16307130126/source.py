import torch
import torch.nn as nn
import torch.nn.functional as F
import fastNLP
from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Tester
from fastNLP.modules import encoder
from fastNLP import Const
from fastNLP import Batch
from fastNLP import BucketSampler
from fastNLP.models import CNNText
#from fastNLP.models import
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import string
from fastNLP import EarlyStopCallback

import re


class RNN(nn.Module):
	def __init__(self, vocab_size, class_size):
		super(RNN, self).__init__()
		self.hidden_size = 256
		# self.num_layer = 1
		self.input_size = 128
		# self.dropout = 0.2
		# self.drop = nn.Dropout(self.dropout)
		self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
		#self.dropout = nn.Dropout(dropout)
		self.embeddings = nn.Embedding(vocab_size, self.input_size)
		self.line = nn.Linear(self.hidden_size, class_size)

	def forward(self, words):
		# words = words.cuda()
		batch_size, _ = words.size()
		z = self.embeddings(words)
		out, (h, c) = self.rnn(z)
		# print(out.size())
		# print(h.size())
		# h = h.permute(1,0,2).contiguous().view(batch_size,-1)
		output = out.sum(dim=1)
		#output = self.dropout(output)
		# output = output.squeeze()
		y = self.line(output)
		# print(y.size())
		return {'pred': y}


class CNN(nn.Module):
	def __init__(self, vocab_size, class_size,dropout=0.1):
		super(CNN, self).__init__()
		self.embedding_size = 256
		self.embedding = nn.Embedding(vocab_size, self.embedding_size)
		# self.convs = nn.ModuleList([nn.Conv1d])
		#self.kernel_sizes = [(3, self.embedding_size), (4, self.embedding_size), (5, self.embedding_size)]
		self.kernel_sizes = [3,4,5]
		self.kernel_nums = [25,25,25]
		'''
		self.conv_pool = encoder.ConvMaxpool(in_channels=self.embedding_size,
            out_channels=self.kernel_nums,
            kernel_sizes=self.kernel_sizes)
        '''

		#self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=kn, kernel_size=ks) for kn, ks in
		#                            zip(self.kernel_nums, self.kernel_sizes)])
		self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_size,out_channels=oc, kernel_size=ks)
			for oc, ks in zip(self.kernel_nums, self.kernel_sizes)])

		'''
		self.ken = []
		for kn,ks in zip(self.kernel_nums,self.kernel_sizes):
			for i in range(kn):
				self.ken.append(ks)
		'''
		'''
		self.convs = nn.ModuleList([nn.Conv1d(
                in_channels=self.embedding_size,
                out_channels=oc,
                kernel_size=ks)
                for oc, ks in zip(self.kernel_nums, self.kernel_sizes)])
        '''
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

		# self.pool = nn.MaxPool1d()
		self.line = nn.Linear(sum(self.kernel_nums), class_size)


	def forward(self, words):
		x = self.embedding(words)
		#x = x.unsqueeze(1)

		x = torch.transpose(x, 1, 2)
		xs = [self.activation(conv(x)) for conv in self.convs]
		xs = [F.max_pool1d(input=i, kernel_size=i.size(2)).squeeze(2) for i in xs]
		xs = torch.cat(xs, dim=-1)
		'''
		xs = [self.activation(conv(x)) for conv in self.convs]
		xs = [i.squeeze(3) for i in xs]
		xs = [F.max_pool1d(input=i,kernel_size=ks[0]) for i,ks in zip(xs,self.ken)]
		xs = [i.sum(dim=2).squeeze(1) for i in xs]
		xs = torch.cat(xs, dim=-1)
		'''
		x = self.dropout(xs)
		x = self.line(x)
		return {'pred': x}


def get_text_classification_datasets():
	categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
	dataset_train = fetch_20newsgroups(subset='train',  data_home='../../..')
	dataset_test = fetch_20newsgroups(subset='test',  data_home='../../..')
	print("In training dataset:")
	print('Samples:', len(dataset_train.data))
	print('Categories:', len(dataset_train.target_names))
	print("In testing dataset:")
	print('Samples:', len(dataset_test.data))
	print('Categories:', len(dataset_test.target_names))
	return dataset_train, dataset_test


def preprocess(input):
	data = input.data
	target = input.target
	dataset = DataSet()
	for i in range(len(data)):
		data_tmp = data[i]
		for c in string.whitespace:
			data_tmp = data_tmp.replace(c, ' ')
		for c in string.punctuation:
			data_tmp = data_tmp.replace(c, '')
		data_tmp = data_tmp.lower().split()
		# print(data_tmp)
		dataset.append(Instance(sentence=data_tmp, target=int(target[i])))
	dataset.apply(lambda x: len(x['sentence']), new_field_name='seq_len')
	return dataset


if __name__ == "__main__":
	model_type = input("RNN or CNN?(r/c)")
	print('initial dataset')
	dataset_train, dataset_test = get_text_classification_datasets()
	print('preprocess dataset')
	train_data = preprocess(dataset_train)
	test_data = preprocess(dataset_test)

	print('set vocabulary')
	vocab = Vocabulary(min_freq=10).from_dataset(train_data, field_name='sentence')
	vocab.index_dataset(train_data, field_name='sentence', new_field_name='words')
	vocab.index_dataset(test_data, field_name='sentence', new_field_name='words')
	print('Vocabulary size=',len(vocab))

	train_data.rename_field('words', Const.INPUT)
	train_data.rename_field('seq_len', Const.INPUT_LEN)
	train_data.rename_field('target', Const.TARGET)
	train_data.set_input(Const.INPUT, Const.INPUT_LEN)
	train_data.set_target(Const.TARGET)

	test_data.rename_field('words', Const.INPUT)
	test_data.rename_field('seq_len', Const.INPUT_LEN)
	test_data.rename_field('target', Const.TARGET)
	test_data.set_input(Const.INPUT, Const.INPUT_LEN)
	test_data.set_target(Const.TARGET)

	print('train...')
	categories = len(dataset_train.target_names)
	traindata, dev_data = train_data.split(0.1)
	print('train data size=',len(traindata))
	print('dev data size=', len(dev_data))
	model = None

	# model = torch.load("model/best_CNN_accuracy_2019-06-02-00-45-56")
	# model = model.cuda()

	if model_type == 'r':
		model = RNN(len(vocab), categories)
		model = model.cuda()

	elif model_type == 'c':
		model = CNN(len(vocab), categories,dropout=0.1)
		model = model.cuda()

	trainer = Trainer(model=model, train_data=traindata, dev_data=dev_data, loss=CrossEntropyLoss(),
	                  metrics=AccuracyMetric(), save_path="model", batch_size=10, n_epochs=20, device='cuda',callbacks=[EarlyStopCallback(10)])
	trainer.train()

	print('test...')
	tester = Tester(test_data, model, metrics=AccuracyMetric(),batch_size=10,device='cuda')
	tester.test()
