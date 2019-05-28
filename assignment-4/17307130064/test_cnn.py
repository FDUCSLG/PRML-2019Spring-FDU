from create_dataset import create_dataset

import torch
from fastNLP import AccuracyMetric
from fastNLP import Tester


vocab, train_data, dev_data, test_data = create_dataset()

model = torch.load('./best_cnn_accuracy_2019-05-28-22-30-20')

metrics = AccuracyMetric(pred='pred', target='target')

tester = Tester(data=test_data, model=model, metrics=metrics)
tester.test()