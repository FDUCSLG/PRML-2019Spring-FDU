import os
os.sys.path.append('../../../assignment-2')

import get_data
from handout import Dataset, get_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
import RNN
import CNN
import string
import re
import numpy as np
from matplotlib import pyplot as plt
import fastNLP
from Padder import AutoPadder_wrapper
from fastNLP import Vocabulary
from fastNLP import Instance
from fastNLP import DataSet
from fastNLP import Trainer
from fastNLP import Tester
from fastNLP import CrossEntropyLoss
from fastNLP import Adam
from fastNLP import AccuracyMetric
from fastNLP.io.model_io import ModelSaver
from fastNLP.io.model_io import ModelLoader
import fitlog
from fastNLP.core.callback import FitlogCallback
import math


def split_text(data):
    text = data['text']
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub('[' + string.whitespace + '\u200b]+', ' ', text)
    text = text.strip().lower()
    list = text.split(' ')
    return list


if __name__ == "__main__":
    fitlog.commit(__file__)             # auto commit your codes
    fitlog.add_hyper_in_file (__file__) # record your hyperparameters
    dataset_train, dataset_test = get_data.get_text_classification_datasets()

    train_set = DataSet()
    for data, label in zip(dataset_train.data, dataset_train.target):
        train_set.append(Instance(text=data, label=int(label)))
    train_set.apply(split_text, new_field_name="text")
    # train_set = train_set[:100]

    test_set = DataSet()
    for data, label in zip(dataset_test.data, dataset_test.target):
        test_set.append(Instance(text=data, label=int(label)))
    test_set.apply(split_text, new_field_name="text")

    vocab = Vocabulary(min_freq=10)
    train_set.apply(lambda x: [vocab.add(word) for word in x['text']])
    test_set.apply(lambda x: [vocab.add(word) for word in x['text']])
    vocab.build_vocab()

    train_set.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name="text")
    test_set.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name="text")

    train_set.set_input("text")
    train_set.set_target("label")
    test_set.set_input("text")
    test_set.set_target("label")

    train_set.set_padder('text', AutoPadder_wrapper())
    test_set.set_padder('text', AutoPadder_wrapper())

    train_set, dev_set = train_set.split(0.1)
    n = train_set.get_length()
    m = len(vocab)
    k = len(dataset_train.target_names)
    print("Size of Train Set:", n)
    print("Total Number of Words:", m)

    rnn_text_model = RNN.RNN_Text(vocab_size=m, input_size=50, hidden_layer_size=128, target_size=k, dropout=0.1)
    cnn_text_model = CNN.CNN_Text(vocab_size=m, input_size=50, target_size=k, dropout=0.05)
    model = rnn_text_model
    # ModelLoader.load_pytorch(model, "model_ckpt_large_CNN.pkl")

    trainer = Trainer(
        train_data=train_set,
        model=model,
        loss=CrossEntropyLoss(pred='pred', target='label'),
        n_epochs=50,
        batch_size=16,
        metrics=AccuracyMetric(pred='pred', target='label'),
        dev_data=dev_set,
        optimizer=Adam(lr=1e-3),
        callbacks=[FitlogCallback(data=test_set)]
    )
    trainer.train()

    # saver = ModelSaver("model_ckpt_large_RNN.pkl")
    # saver.save_pytorch(model)

    tester = Tester(
        data=train_set,
        model=model,
        metrics=AccuracyMetric(pred='pred', target='label'),
        batch_size=16,
    )
    tester.test()

    tester = Tester(
        data=test_set,
        model=model,
        metrics=AccuracyMetric(pred='pred', target='label'),
        batch_size=16,
    )
    tester.test()

    fitlog.finish()                     # finish the logging
