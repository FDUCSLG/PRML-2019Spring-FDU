from fastNLP import DataSet, Instance, Vocabulary, Const
from fastNLP import CrossEntropyLoss, AccuracyMetric, Trainer, Tester
from fastNLP.models import CNNText
from models import RNNText
# from models.custom_rnn import RNNText

from preprocess import get_train_dev_test_vocab

if __name__ == '__main__':

    train_data, dev_data, test_data, vocab = get_train_dev_test_vocab()

    model_rnn = RNNText(vocab_size=len(vocab),
                         embedding_dim=32, 
                         hidden_size=10,
                         output_size=5)

    loss = CrossEntropyLoss(pred=Const.OUTPUT, 
                            target=Const.TARGET)

    metrics = AccuracyMetric(pred=Const.OUTPUT, 
                             target=Const.TARGET)

    trainer = Trainer(model=model_rnn,
                      train_data=train_data, 
                      dev_data=dev_data, 
                      loss=loss, 
                      metrics=metrics)

    trainer.train()

    tester = Tester(test_data, 
                    model_cnn, 
                    metrics=AccuracyMetric())

    tester.test()

    



