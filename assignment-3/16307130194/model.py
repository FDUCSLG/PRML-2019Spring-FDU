import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size):
        super(PoetryModel, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=2)
        self.linear = nn.Linear(self.hidden_size, vocabulary_size)

    def forward(self, input, hidden_state=None):
        sequence_length, batch_size = input.size()
        if hidden_state is None:
            h = input.data.new(2, batch_size, self.hidden_size).fill_(0).float()
            c = input.data.new(2, batch_size, self.hidden_size).fill_(0).float()
        else:
            h, c = hidden_state
        # size: [sequence_length, batch_size, embedding_size]
        embeddings = self.embeddings(input)
        # output size: [sequence_length, batch_size, hidden_size]
        output, hidden = self.lstm(embeddings, (h, c))

        # size: [sequence_length * batch_size, vocabulary_size]
        output = self.linear(output.view(sequence_length * batch_size, -1))
        return output, hidden


class LSTM(object):
    def __init__(self):
        pass
