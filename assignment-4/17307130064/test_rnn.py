from create_dataset import create_dataset

import torch
from fastNLP import AccuracyMetric
from fastNLP import Tester


vocab, train_data, dev_data, test_data = create_dataset()

model = torch.load('./best_lstm_accuracy_2019-05-29-00-31-49')

metrics = AccuracyMetric(pred='pred', target='target')

tester = Tester(data=test_data, model=model, metrics=metrics, device=0)
tester.test()