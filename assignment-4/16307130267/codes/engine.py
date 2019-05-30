import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import random
import models
import utility
from matplotlib import pyplot as plt

SEED = 233
torch.manual_seed(SEED)

class Engine(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_iterator, self.valid_iterator, self.test_iterator =\
            self.init_data_iterator()
        
        self.model, self.optimizer = self.init_model_optimizer()

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = self.criterion.to(self.device)

    def init_data_iterator(self):
        if self.args['model']['arch'] == 'FastText':
            self.TEXT = data.Field(preprocessing=utility.generate_bigrams)
        else:
            self.TEXT = data.Field()
        self.LABEL = data.LabelField(dtype=torch.float)
        fields = {'label': ('label', self.LABEL), 'text': ('text', self.TEXT)}
        train_data, test_data = data.TabularDataset.splits(
                                    path = self.args['data']['dataset_path'],
                                    train = 'train.json',
                                    test = 'test.json',
                                    format = 'json',
                                    fields = fields)
        train_data, valid_data = train_data.split(random_state=random.seed(SEED))

        self.TEXT.build_vocab(train_data, max_size=25000)
        self.LABEL.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = self.args['data']['batch_size'],
            )
        return train_iterator, valid_iterator, test_iterator

    def init_model_optimizer(self):
        model = models.__dict__[self.args['model']['arch']](
            len(self.TEXT.vocab),
            self.args['model']['params'])
        model = model.to(self.device)

        optimizer = optim.__dict__[self.args['model']['optimizer']](
            params=model.parameters(),
            lr=self.args['model']['learning_rate'],
            weight_decay=self.args['model']['weight_decay'])
        return model, optimizer

    def load_model(self):
        ckpt = torch.load(self.args['model']['path']+self.args['name']+'.ckpt')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        #self.model = torch.load(self.args['model']['path']+self.args['name']+'.pt')
        print('Model loaded.')
        
    def save_model(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.args['model']['path']+self.args['name']+'.ckpt')
        print('Model saved.')

    def load_pretrained_model(self):
        print('Loading pre-trained model ...')
        pre_train = gensim.models.KeyedVectors.load_word2vec_format(self.args['data']['pretrain_path'])
        weights = torch.FloatTensor(pre_train.vectors)
        self.model.embedding.weight = torch.nn.Parameter(weights)
        print('Finish loading.')

    def accuracy(self, preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train_step(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in iterator:
            self.optimizer.zero_grad()
            predictions = self.model(batch.text).squeeze(1)
            loss = self.criterion(predictions, batch.label)
            acc = self.accuracy(predictions, batch.label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_step(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = self.model(batch.text).squeeze(1)
                loss = self.criterion(predictions, batch.label)
                acc = self.accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self):
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        print('Begin training...')
        for epoch in range(self.args['n_epoch']):
            train_loss, train_acc = self.train_step(self.train_iterator)
            valid_loss, valid_acc = self.evaluate_step(self.valid_iterator)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_acc.append(valid_acc)
            print('Epoch: ', epoch+1, \
                'Train Loss is:', train_loss, \
                'Train Acc is:', train_acc, \
                'Val Loss is:', valid_loss, \
                'Val Acc is: ', valid_acc
                )
        self.plot()

    def test(self):
        test_loss, test_acc = self.evaluate_step(self.test_iterator)
        self.test_loss = test_loss
        self.test_acc = test_acc
        print('Test Loss is:', test_loss, \
            'Test Acc is: ', test_acc)

    def predict(self, sentence, min_len=5):
        from spacy.lang.zh import Chinese
        import spacy
        nlp = Chinese()

        tokenized = [tok.text for tok in nlp(sentence)]
        
        if self.args['model']['arch'] == 'CNN':
            if len(tokenized) < min_len:
                tokenized += ['<pad>'] * (min_len - len(tokenized))
        
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(self.model(tensor))
        print(prediction.item())

    def plot(self):
        nums = len(self.train_loss)
        x = range(nums)
        plt.suptitle('loss and accuracy curves in training procedure', fontsize=16)
        plt.title('model: %s, optimizer: %s, epochs: %d, learning rate: %f' % 
                 (self.args['model']['arch'],
                  self.args['model']['optimizer'],
                  self.args['n_epoch'], 
                  self.args['model']['learning_rate']), fontsize=10)
        plt.plot(x, self.train_loss, label='train loss', color='#FFA500')
        plt.plot(x, self.train_acc, label='train acc', color='#00FFCC')
        plt.plot(x, self.valid_loss, label='valid loss', color='#CC9933')
        plt.plot(x, self.valid_acc, label='valid acc', color='#00CC99')
        plt.legend()
        plt.show()