from data_processing import *
from fastNLP.models import CNNText
from fastNLP import CrossEntropyLoss
from fastNLP import Adam
from fastNLP import AccuracyMetric
from fastNLP import Trainer, Tester
import fastNLP
import torch.nn as nn
import torch.nn.functional as F
class myCNNText(nn.Module):
    """
    Text classification model by CNN, the implementation of paper
    """
    def __init__(self, embed_num, embed_dim, num_classes, kernel_nums=(100, 100, 100), kernel_sizes=(3,4,5), padding=0, dropout=0.5, pre_weight=None):
        super(myCNNText, self).__init__()
        if pre_weight is None:
            self.embed = nn.Embedding(embed_num, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(pre_weight)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_nums[i], (kernel_sizes[i], embed_dim)) for i in range(len(kernel_sizes))])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, word_seq):
        """
        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(word_seq)  # [N, L] -> [N, L, C]
        x = x.unsqueeze(1)
        x = [self.conv_and_pool(x, conv) for conv in self.convs1]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return {'pred': x}

    def predict(self, word_seq):
        output = self(word_seq)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}

"""
train_data, test_data, vocab = get_fastnlp_dataset()
weight = get_pretrained_weight(vocab)
cnn_text_model = myCNNText(embed_num=len(vocab), embed_dim=50, num_classes=8, padding=2, dropout=0.1, pre_weight=weight)
trainer = Trainer(train_data=train_data, model=cnn_text_model,
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



new_model = myCNNText(embed_num=len(vocab), embed_dim=50, num_classes=8, padding=2, dropout=0.1, pre_weight=weight)
new_model = load_model(new_model, './model/best_myCNNText_acc_2019-05-27-11-12-48')

tester = Tester(test_data, new_model, metrics=AccuracyMetric())
eval_results = tester.test()
print(eval_results)

"""
