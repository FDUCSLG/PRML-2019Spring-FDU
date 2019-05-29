import torch
import torch.nn as nn


class TextCNN(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.embedding = nn.Embedding(config.vocab_s, config.embedding_s)
    self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=1,
                                    out_channels=config.feat_s,
                                    kernel_size=(h, config.embedding_s)),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=config.max_len))
              for h in config.window_s])
    self.dropout = nn.Dropout(config.dropout_rate)
    self.fc = nn.Linear(config.feat_s*len(config.window_s), config.num_class)


  def forward(self, words):  
    embed = self.embedding(words)

    embed = embed.unsqueeze(1)

    out = [conv(embed) for conv in self.convs]

    out = torch.cat(out, dim=1)
    out = out.view(-1, out.size(1))

    out = self.dropout(out)
    out = self.fc(out)

    return {"pred": out}


class TextRNN(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.embedding = nn.Embedding(config.vocab_s, config.embedding_s)
    self.lstm = nn.LSTM(config.embedding_s, config.hidden_s, batch_first=True)
    self.avg1d = nn.AvgPool1d(config.max_len)
    self.dropout = nn.Dropout(config.dropout_rate)
    self.fc = nn.Linear(config.hidden_s, config.num_class)

  
  def forward(self, words):
    embed = self.embedding(words)

    out, _ = self.lstm(embed)

    out = out.permute(0, 2, 1)
    out = self.avg1d(out)

    out = out.view(-1, out.size(1))
    
    out = self.dropout(out)
    out = self.fc(out)

    # print("out shape", out.shape, "\n\n")

    return {"pred": out}

