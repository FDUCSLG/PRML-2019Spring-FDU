import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import pandas as pd
import sys
sys.path.append('.')
from utils import matrix_mul,element_wise_mul

class WordAttentionNet(nn.Module):
    def __init__(self, dict, hidden_size=50):
        super(WordAttentionNet, self).__init__()
        self.dict = dict[:, 1:]
        dict_len, embed_size = self.dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        self.dict = torch.from_numpy(np.concatenate(
            [unknown_word, self.dict], axis=0).astype(np.float))
        self.hidden_size = hidden_size

        self.embeds = nn.Embedding(
            num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(self.dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.weight = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(2 * hidden_size, 1), nn.Tanh())
        # self.word_weight = nn.Parameter(
        #     torch.Tensor(2 * hidden_size, 2 * hidden_size))
        # self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self._init_weight(0, 0.05)

    def _init_weight(self, mean=0, std=0.05):
        nn.init.normal_(self.weight[0].weight,mean, std)
        nn.init.normal_(self.context[0].weight,mean, std)

    def forward(self, input, hidden_state):
        output = self.embeds(input)
        f_output, h_output = self.gru(output.float(), hidden_state)
        output = self.weight(f_output)
        output = self.context(output).squeeze().permute(1, 0)
        output = F.softmax(output,dim=-1)
        output = element_wise_mul(f_output, output.permute(1, 0))
        return output, h_output


class SentAttentionNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=20):
        super(SentAttentionNet, self).__init__()

        self.gru = nn.GRU(2 * word_hidden_size,
                          sent_hidden_size, bidirectional=True)
        self.sent_weight = nn.Parameter(torch.Tensor(
            2 * sent_hidden_size, 2 * sent_hidden_size))
        self.weight = nn.Sequential(
            nn.Linear(2 * sent_hidden_size, 2 * sent_hidden_size), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(2 * sent_hidden_size, 1), nn.Tanh())
        # self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        # self.context_weight = nn.Parameter(
        #     torch.Tensor(2 * sent_hidden_size, 1))
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

        self._init_weight(0, 0.05)

    def _init_weight(self, mean, std):
        nn.init.normal_(self.weight[0].weight,mean, std)
        nn.init.normal_(self.context[0].weight,mean, std)
        nn.init.normal_(self.fc.weight, mean, std)
        nn.init.normal_(self.fc.bias, mean, std)
        

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = self.weight(f_output)
        output = self.context(output).squeeze().permute(1, 0)
        output = F.softmax(output,dim=-1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)
        return output,h_output


class HierNet(nn.Module):
    def __init__(self, dict,batch_size, word_hidden_size=50, sent_hidden_size=50, num_classes=20):
        super(HierNet, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttentionNet(
            dict, word_hidden_size)
        self.sent_att_net = SentAttentionNet(
            sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden(batch_size)

    def _init_hidden(self,batch_size):
        self.word_hidden_state = torch.zeros(
            2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(
            2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):
        output_list = []
        input = input.permute(1,0,2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(
                i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(
            output, self.sent_hidden_state)

        return output


if __name__ == '__main__':
    dict = pd.read_csv(filepath_or_buffer='./data/glove.6B.50d.txt',
                       header=None, sep=" ", quoting=csv.QUOTE_NONE).values
    model = HierNet(dict,2).cuda()
    print(model)
    input = torch.randint(len(dict), (2, 10, 5)).cuda()
    output = model(input)
    print(output)
