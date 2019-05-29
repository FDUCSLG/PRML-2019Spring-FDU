import numpy as np
from torch import nn
from torch.optim import SGD
from fastNLP import Trainer
from fastNLP import DataSet, Vocabulary, Instance
from fastNLP import AccuracyMetric, CrossEntropyLoss
import torch
import string
import argparse
from sklearn.datasets import fetch_20newsgroups
from my_model import rnn, cnn, myLSTM


def get_text_classification_datasets(n_class):
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
                  'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                  'talk.politics.misc', 'talk.religion.misc']
    dataset_train = fetch_20newsgroups(subset='train', categories=categories[0: n_class], data_home='../../..')
    return dataset_train


def remove_punc(s):
    for i in s:
        if i in string.punctuation or i in string.whitespace:
            s = s.replace(i, " ")
    return s


def handle_data(n_class):
    train_data = get_text_classification_datasets(n_class)
    dataset = DataSet()
    vocab = Vocabulary(min_freq=0, unknown='<unk>', padding='<pad>')
    for i in range(len(train_data.data)):
        ans = remove_punc(train_data.data[i])
        dataset.append((Instance(content=ans, target=int(train_data.target[i]))))
    dataset.apply(lambda x: x['content'].lower().split(), new_field_name='words', is_input=True)
    for txt in dataset:
        vocab.add_word_lst(txt['words'])
    vocab.build_vocab()
    # index句子, Vocabulary.to_index(word)
    dataset.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='index')
    dataset.set_input("index")
    dataset.set_target("target")
    tra, dev = dataset.split(0.2)
    return tra, dev, len(vocab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", "-m", default="lstm", choices=["rnn", "lstm", "cnn"])
    parser.add_argument("--n_epochs", "-n", default=5, type=int)
    parser.add_argument("--embedding", "-e", default=100, type=int)
    parser.add_argument("--category", "-c", default=4, type=int)
    parser.add_argument("--batch", "-b", default=4, type=int)
    parser.add_argument("--learning_rate", "-l", default=0.005, type=float)
    args = parser.parse_args()
    if args.category > 20 or args.category < 1:
        raise Exception("the number of category must be between 1 and 20")
    train_data, test_data, dic_size= handle_data(args.category)
    if args.methods == "rnn":
        model = rnn(dic_size, args.category)
        output = "rnn_model.pth"
    elif args.methods == "lstm":
        model = myLSTM(dic_size, args.category)
        output = "lstm_model.pth"
    else:
        #model = cnn(dic_size, args.category)
        model = torch.load("cnn_model.pth")
        output = "cnn_model.pth"
    trainer = Trainer(train_data, model, loss=CrossEntropyLoss(pred="pred", target='target'),
                      optimizer=SGD(model.parameters(), lr=args.learning_rate), n_epochs=args.n_epochs,
                      dev_data=test_data, metrics=AccuracyMetric(pred="pred", target='target'), batch_size=args.batch)
    trainer.train()
    torch.save(model, output)


if __name__ == "__main__":
    main()