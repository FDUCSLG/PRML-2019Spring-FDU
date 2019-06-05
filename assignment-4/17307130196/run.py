from model.CNNclassifier import *
from model.RNNclassifier import *
from data.preprocess import *
from fastNLP import AccuracyMetric
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import RMSprop
from fastNLP import Tester
from data.assis import MyCallback

loss=CrossEntropyLoss()
metrics=AccuracyMetric(target='label',pred='output')
callback=MyCallback()
D_dict=2001
D_H=128
D_input=128
D_output=4

def RNNmethod():
    train_data,dev_data,test_data,D_dict,vocab=data_preprocess()
    LSTM=LSTMText(D_input,D_H,D_output,D_dict)
    print('start train')
    optim=RMSprop(LSTM.parameters(),lr=0.01)
    tester = Tester(train_data, LSTM, metrics)

    trainer = Trainer(model=LSTM, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics,batch_size=32,optimizer=optim,save_path='.record/',validate_every=30,callbacks=[callback])
    # trainer = Trainer(model=LSTM, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics,batch_size=32,optimizer=optim,save_path='.record/',validate_every=30)
    while 1:
        trainer.train()
        input('You stop the program')
        tester = Tester(test_data, LSTM, metrics)
        tester.test()
        tester = Tester(train_data, LSTM, metrics)
        tester.test()
        tester = Tester(dev_data, LSTM, metrics)
        tester.test()


def CNNmethod():
    train_data,dev_data,test_data,D_dict,vocab=data_preprocess()
    CNN=CNNText(D_input,D_output,D_dict=D_dict)
    print('start train')
    optim=RMSprop(CNN.parameters(),lr=0.01)
    
    trainer = Trainer(model=CNN, train_data=train_data, dev_data=train_data, loss=loss, metrics=metrics,batch_size=32,optimizer=optim, save_path='.record/', validate_every=30,callbacks=[callback])

    while 1:
        trainer.train()
        input('You stop the program')
        tester = Tester(test_data, LSTM, metrics)
        tester.test()
        tester = Tester(train_data, LSTM, metrics)
        tester.test()
        tester = Tester(dev_data, LSTM, metrics)
        tester.test()



def test():
    train_data,dev_data,test_data,D_dict,vocab=data_preprocess()
    # test_set=test_data(vocab)
    LSTM=LSTMText(D_input,D_H,D_output,D_dict=D_dict)
    LSTM.load_state_dict(torch.load('.record/rnn record').state_dict())
    
    tester1 = Tester(test_data, LSTM, metrics)
    tester1.test()
    print('This one')
    tester2 = Tester(train_data, LSTM, metrics)
    tester2.test()
    print('This one')
    tester3 = Tester(dev_data, LSTM, metrics)
    tester3.test()
    print('This one')
    test_set=test_data_(vocab)
    tester4 = Tester(test_set, LSTM, metrics)
    tester4.test()

# test()

CNNmethod()
# RNNmethod()