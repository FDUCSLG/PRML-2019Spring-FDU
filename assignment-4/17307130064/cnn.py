import torch.nn as nn
import fastNLP.modules.encoder as encoder


class cnn(nn.Module):

    def __init__(self, init_embed, num_classes, kernel_nums=(3, 4, 5), kernel_sizes=(3, 4, 5), padding=0, dropout=0.5):
        super(cnn, self).__init__()

        self.embed = encoder.Embedding(init_embed)
        self.conv_pool = encoder.ConvMaxpool(in_channels=self.embed.embedding_dim, out_channels=kernel_nums,
                                             kernel_sizes=kernel_sizes, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, words, seq_len=None):
        x = self.embed(words)
        x = self.conv_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return {'pred' : x}

    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output['pred'].max(dim=1)
        return {'pred' : predict}