import torch
import torch.nn as nn
from fastNLP.modules import encoder

class CNN(nn.Module):
	def __init__(self, vocab_size,
					   embed_dim, num_classes, 
					   padding=0, dropout=0.5,
					   kernel_nums=(50, 50, 50), 
					   kernel_sizes=(3, 4, 5)):
	
		super(CNN, self).__init__()
		self.embed_dim = embed_dim
		self.embedding = nn.Embedding(vocab_size, self.embed_dim)
		self.conv_pool = encoder.ConvMaxpool(in_channels=self.embed_dim,
											 out_channels=kernel_nums,
											 kernel_sizes=kernel_sizes,
											 padding=padding
											 )
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(sum(kernel_nums), num_classes)

	def forward(self, input):
		x = self.embedding(input)
		x = self.conv_pool(x)
		x = self.dropout(x)
		x = self.fc(x)
		return {"output":x}

class RNN(nn.Module):
	def __init__(self, vocab_size,
					   embed_dim, num_classes,
					   hidden_dim, num_layers=2, 
					   bidirect=True, dropout=0.5):
	
		super(RNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
							bidirectional=bidirect, dropout=dropout)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_dim * 2, num_classes)

	def forward(self, input):
		input = input.permute(1, 0)
		input = self.embedding(input)
		output, (h, c) = self.lstm(input)
		h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
		h = self.dropout(h)
		output = self.fc(h)
		
		return {"output":output}



 