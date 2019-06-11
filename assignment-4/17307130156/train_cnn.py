from fastNLP import DataSet, Instance, Vocabulary, Const
from fastNLP import CrossEntropyLoss, AccuracyMetric, Trainer, Tester
from fastNLP import Callback
from models import CNNText
from models import RNNText

from preprocess import get_train_dev_test_vocab

checkpoint_path = 'checkpoints/cnn_t.pt'

model = None

class MyCallback(Callback):
    def on_step_end(self):
        torch.save(model, checkpoint_path)
        print ('Save model')

def train():

    train_data, dev_data, test_data, vocab = get_train_dev_test_vocab()

    model = CNNText(vocab_size=len(vocab),
                        embedding_dim=50, 
                        output_size=20)
    
    loss = CrossEntropyLoss(pred=Const.OUTPUT, 
                            target=Const.TARGET)

    metrics = AccuracyMetric(pred=Const.OUTPUT, 
                             target=Const.TARGET)

    trainer = Trainer(model=model,
                      train_data=train_data, 
                      dev_data=dev_data, 
                      loss=loss, 
                      metrics=metrics, 
                      n_epochs=100)

    trainer.train()

    tester = Tester(test_data, 
                    model, 
                    metrics=AccuracyMetric())

    tester.test()

    

if __name__ == '__main__':
    train()
