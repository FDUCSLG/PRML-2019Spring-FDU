import torch
import torch.nn as nn


class RNNText(nn.Module):

    def __init__(self,
                 vocab_size, 
                 embedding_dim, 
                 output_size, 
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.5):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=True,
                            dropout=dropout)

        self.linear = nn.Linear(hidden_size * 2, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        # words : (batch_size, seq_len)

        words = words.permute(1,0)
        # (seq_len, batch_size)

        embedded = self.dropout(self.embedding(words))
        # embedded : (seq_len, batch_size, embedding_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # output: (seq_len, batch_size, hidden_size * 2)
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # cell: (num_layers * 2, batch_size, hidden_size)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # hidden: (batch_size, hidden_size * 2)

        pred = self.linear(hidden.squeeze(0))
        # result: (batch_size, output_size)
        return {"pred":pred}




