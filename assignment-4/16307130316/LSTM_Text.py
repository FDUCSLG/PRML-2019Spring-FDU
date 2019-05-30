from data_processing import *
import torch
import torch.nn as nn
from fastNLP import CrossEntropyLoss
from fastNLP import Adam
from fastNLP import AccuracyMetric
from fastNLP import Trainer, Tester
import fastNLP

class myRNNText(nn.Module):
    """
    Text classification model by RNN, the implementation of paper
    """
    def __init__(self, embed_num,
                 embed_dim,
                 num_classes,
                 hidden_dim,
                 num_layer,
                 bidirectional,
                 pre_weight=None
                 ):
        super(myRNNText, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.bidirectional = bidirectional

        # ready to use pre-trained embedding?
        if pre_weight is None:
            self.embed = nn.Embedding(embed_num, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(pre_weight)
        # self.embed = encoder.Embedding(embed_num, embed_dim)

        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, self.num_layer, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim*2, num_classes) if self.bidirectional else nn.Linear(self.hidden_dim, num_classes)

    def forward(self, word_seq):
        """
        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(word_seq)
        h0 = torch.zeros(self.num_layer*2, x.size(0), self.hidden_dim) if self.bidirectional \
            else torch.zeros(self.num_layer, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layer*2, x.size(0), self.hidden_dim) if self.bidirectional \
            else torch.zeros(self.num_layer, x.size(0), self.hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = torch.mean(out, dim=1, keepdim=True)
        out = self.fc(out[:, -1, :])
        return {'pred': out}

    def predict(self, word_seq):
        """
        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return predict: dict of torch.LongTensor, [batch_size, seq_len]
        """
        output = self(word_seq)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}



"""
train_data, test_data, vocab = get_fastnlp_dataset()
weight = get_pretrained_weight(vocab)
rnn_text_model = myRNNText(embed_num=len(vocab), embed_dim=128, num_classes=8, hidden_dim=128, num_layer=1, bidirectional=False)
rnn_text_model = load_model(rnn_text_model, './model/best_myRNNText_acc_2019-05-28-00-24-36')
trainer = Trainer(train_data=train_data, model=rnn_text_model,
                  loss=CrossEntropyLoss(pred='pred', target='target'),
                  metrics=AccuracyMetric(),
                  n_epochs=10,
                  batch_size=32,
                  print_every=-1,
                  validate_every=-1,
                  dev_data=test_data,
                  save_path='./model',
                  optimizer=Adam(lr=1e-3, weight_decay=0),
                  check_code_level=-1,
                  metric_key='acc',
                  use_tqdm=False,
                  )
trainer.train()

new_model = myRNNText(embed_num=len(vocab), embed_dim=128, num_classes=8, hidden_dim=128, num_layer=1, bidirectional=False)
new_model = load_model(new_model, './model/best_myRNNText_acc_2019-05-28-15-14-33')

tester = Tester(test_data, new_model, metrics=AccuracyMetric())
eval_results = tester.test()
print(eval_results)

"""
