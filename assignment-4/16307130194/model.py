import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_num, kernel_sizes,
                 dropout, static, in_channels):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = kernel_num
        self.static = static

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, input):
        x = self.embed(input)
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.fc(x)
        return {"output": output}


class CNN_w2v(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_num, kernel_sizes,
                 dropout, static, in_channels, weight):
        super(CNN_w2v, self).__init__()
        self.in_channels = in_channels
        self.out_channels = kernel_num
        self.static = static

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.convs = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, input):
        x = self.embed(input)
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.fc(x)
        return {"output": output}


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.permute(1, 0)
        embeds = self.embedding(input)
        embeds = self.dropout(embeds)
        # self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(embeds)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)

        output = self.fc(hidden.squeeze(0))
        return {"output": output}


class LSTM_maxpool(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.permute(1, 0)
        embeds = self.embedding(input)
        embeds = self.dropout(embeds)
        # self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(embeds)

        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        pool = F.max_pool1d(output.permute(1, 2, 0), output.size(0)).squeeze(2)
        # hidden = self.dropout(hidden)
        pool = self.dropout(pool)

        # output = self.fc(hidden.squeeze(0))
        output = self.fc(pool)
        return {"output": output}


class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(2 * hidden_dim + embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = input.permute(1, 0, 2)
        embeds = self.embedding(input)
        embeds = embeds.permute(1, 0, 2)
        # embeds = self.dropout(embeds)
        # self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(embeds)

        output = torch.cat((output, embeds), 2)
        output = output.permute(1, 0, 2)
        output = self.linear(output).permute(0, 2, 1)

        pool = F.max_pool1d(output, output.size(2)).squeeze(2)
        # hidden = self.dropout(hidden)
        # pool = self.dropout(pool)

        # output = self.fc(hidden.squeeze(0))
        output = self.fc(pool)
        return {"output": output}


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.permute(1, 0)
        embeds = self.embedding(input)
        embeds = self.dropout(embeds)
        output, hidden = self.rnn(embeds)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)

        output = self.fc(hidden.squeeze(0))
        return {"output": output}
