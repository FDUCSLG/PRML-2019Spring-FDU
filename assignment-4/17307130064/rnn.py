import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_length, hidden_size, output_size):
        super(lstm, self).__init__()

        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, words, seq_len=None):
        input = self.word_embeddings(words.cuda())
        input = input.permute(1, 0, 2)
        h_0 = torch.zeros(1, int(input.shape[1]), self.hidden_size).cuda()
        c_0 = torch.zeros(1, int(input.shape[1]), self.hidden_size).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])

        return {'pred' : final_output}