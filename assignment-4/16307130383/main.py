from model import RNN, CNN
from preprocess import build_dataset
from fastNLP import Trainer
from fastNLP import Adam
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
import torch

# categories = ['comp.graphics',
#               'comp.os.ms-windows.misc',
#               'comp.sys.mac.hardware',
#               'misc.forsale',
#               'rec.motorcycles',
#               'rec.sport.baseball',
#               'sci.crypt',
#               'sci.electronics',
#               'sci.space',
#               'soc.religion.christian',
#               'talk.politics.guns',
#               'talk.politics.mideast',
#               'talk.religion.misc']
categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]      
vocab, train_set, test_set = build_dataset(train_size=2000, test_rate=0.1, categories=categories)
vocab_size = len(vocab)
input_dim = 256
hidden_dim = 256
output_dim = len(categories)
in_channels = 1
out_channels = 256
kernel_sizes = [3, 4, 5]
keep_proba = 0.5
print('vocab size: ', vocab_size, '  output_dim: ', output_dim)
print('train size: ', len(train_set), '  test size: ', len(test_set))

# model = RNN(vocab_size, input_dim, hidden_dim, output_dim)
model = CNN(vocab_size, input_dim, output_dim, in_channels, out_channels, kernel_sizes, keep_proba)

def train(epochs=10, lr=0.001):
  trainer = Trainer(model=model, train_data=test_set, dev_data=test_set,
                    loss=CrossEntropyLoss(pred='output', target='target'),
                    metrics=AccuracyMetric(pred='pred', target='target'),
                    optimizer=Adam(lr=lr),
                    save_path='../model',
                    batch_size=1,
                    n_epochs=epochs)
  trainer.train()

def get_model():
  return model

def save(path):
  torch.save(model.state_dict(), path)

def load(path):
  model.load_state_dict(torch.load(path))
  model.eval()