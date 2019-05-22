import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, n_classes=20, input_length=1014, input_dim=68, n_conv_filters=256, n_fc_neurons=1024, dropout_prob=0.5, init_mean=0, init_std=0.05):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(
            input_dim, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 17 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(dropout_prob))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons/2), nn.Dropout(dropout_prob))
        self.fc3 = nn.Linear(n_fc_neurons / 2, n_classes)

        self._init_weight(init_mean, init_std)

    def _init_weight(self, mean, std):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, mean, std)

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
