import os
os.sys.path.append('..')
from handout import *
import fastNLP
from fastNLP import DataSet
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Optimizer
from fastNLP import core
import data
import model

CNN = False
RNN = True

if __name__ == '__main__':
    if CNN:
        d_train,d_test = data.getCharDataset()
        m = model.CharacterLevelCNN()
        ADAMOP = fastNLP.Adam(lr=0.001,weight_decay=0,betas=(0.9,0.999))
        trainner = Trainer(
            train_data=d_train,
            model=m,
            n_epochs=100,
            batch_size=128,
            use_cuda=True,
            check_code_level=0,
            optimizer=ADAMOP,
            dev_data=d_test,
            metrics=core.metrics.AccuracyMetric(target="label")
        )
        trainner.train()
    if RNN:
        d_train,d_test,embedding = data.getWordDataset()
        m = model.LSTMClassifier(32,20,256,400001,200,embedding)
        ADAMOP = fastNLP.Adam(lr=0.001,weight_decay=0,betas=(0.9,0.999))
        trainner = Trainer(
            train_data=d_train,
            model=m,
            n_epochs=100,
            batch_size=32,
            use_cuda=True,
            check_code_level=-1,
            optimizer=ADAMOP,
            dev_data=d_test,
            metrics=core.metrics.AccuracyMetric(target="label")
        )
        trainner.train()