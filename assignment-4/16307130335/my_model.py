from torch import nn
import torch
import string

class cnn(nn.Module):
    def __init__(self, input_dim, n_class):
        super(cnn, self).__init__()
        vocb_size = input_dim
        self.dim = 100
        self.max_len = 20000
        self.embeding = nn.Embedding(vocb_size, self.dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=4,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=3,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.out = nn.Linear(48, n_class)


    def forward(self, index):
        x = self.embeding(index)
        x = x.unsqueeze(3)
        x = x.permute(0, 2, 1, 3)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, x.size(1))
        output = self.out(x)
        return {"pred": output}


class rnn(nn.Module):
    def __init__(self, input_size, n_class):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=100,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.embedding = nn.Embedding(input_size, 100)
        self.out = nn.Linear(32, n_class)

    def forward(self, index):
        data = self.embedding(index)
        output, hidden = self.rnn(data)
        output = self.out(output)
        # 仅仅获取 time seq 维度中的最后一个向量
        output = torch.mean(output, dim=1, keepdim=True)
        return {"pred": output[:,-1,:]}


class myLSTM(nn.Module):
    def __init__(self, input_size, n_class):
        super().__init__()
        self.myLSTM = torch.nn.LSTM(
            input_size=100,
            hidden_size=16,
            num_layers=1,
            batch_first=True
        )
        self.embedding = nn.Embedding(input_size, 100)
        self.out = nn.Linear(16, n_class)

    def forward(self, index):
        x = self.embedding(index)
        output, hidden = self.myLSTM(x)
        output = torch.mean(output, dim=1, keepdim=True)
        output_in_last_timestep=output[:,-1,:]
        x = self.out(output_in_last_timestep)
        return {"pred": x}

