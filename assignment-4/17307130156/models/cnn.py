import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP import Const

class CNNText(nn.Module):

    def __init__(self, 
                 vocab_size=1000, 
                 embedding_dim=300, 
                 kernel_h=[3, 4, 5], # Three kinds of filter with window of 3, 4, 5 words respectively
                 kernel_num=100, # 100 filters of each kind
                 output_size=2, # number of classes
                 dropout=0.5,

                 pretrained_embeddings=None):

        super(CNNText, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.dropout = nn.Dropout(dropout)

        self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embedding_dim)) for K in kernel_h])

        self.linear = nn.Linear(len(kernel_h) * kernel_num, output_size)

    def forward(self, words, seq_len=None):
        # words: (batch_size, seq_len)

        x = self.embedding(words)  
        # (batch_size, seq_len, embedding_dim)

        x = x.unsqueeze(1)  
        # (batch_size, in_channels=1, height_in=seq_len, width_in=embedding_dim)

        x = [conv(x) for conv in self.conv]
        # [(batch_size, out_channels, height_out, width_out=1), ...]
        # width_out is 1 since width of each kernel is exactly width_in, thus we can squeeze it
        x = [xi.squeeze(3) for xi in x]
        # [(batch_size, out_channels, height_out), ...]
        x = [F.relu(xi) for xi in x]

        # Equivalent to: x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # [(batch_size, out_channels), ...]

        x = torch.cat(x, 1)
        # (batch_size, kernel_types * out_channels)

        x = self.dropout(x)
        x = self.linear(x)
        # (batch_size, output_size)

        return {Const.OUTPUT: x}


if __name__ == '__main__':
    model = CNNText(kernel_h=[1, 2, 3, 4], vocab_size=3, embedding_dim=2)
    x = torch.LongTensor([[1, 2, 1, 2, 0]])
    print(model(x))
