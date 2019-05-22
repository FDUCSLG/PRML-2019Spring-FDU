import numpy as np
import os
import json
import progressbar
import time
from matplotlib import pyplot as plt
from data import get_data_mini, get_data_big
from model import GeneratingLSTM
from loss import temporal_softmax_loss
import optim
import utility

class Engine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.train_data, self.dev_data, self.test_data = self.data_loader()
        self.model, self.optimizer = self.init_model_optimizer()
        self.metric = temporal_softmax_loss
        self.epoch_loss = []
        self.epoch_perp = []

    def data_loader(self, props=[0.6, 0.2, 0.2], drop_last=True, shuffle=False):
        assert len(props) == 3, ':-('
        if self.args['data']['use_mini']:
            data, word2idx, idx2word = get_data_mini(path='dataset/tang_mini.npz')
        else:
            data, word2idx, idx2word = get_data_big(path='dataset/tang_big.npz')
        self.word2idx = word2idx
        self.idx2word = idx2word
        bs = self.args['data']['batch_size']
        nums = len(data) // bs
        # get batch data
        data_batch = []
        for i in range(nums):
            data_batch.append(data[i*bs:(i+1)*bs])
        if not drop_last:
            data_batch.append(data[nums*bs:])
        
        if shuffle:
            indices = list(range(len(data_batch)))
            np.random.shuffle(indices)
            for i in range(len(indices)):
                data_batch[i] = data_batch[indices[i]]

        bounds = [int(len(data_batch) * prop) for prop in props]
        train_data = data_batch[:bounds[0]]
        dev_data = data_batch[bounds[0]:bounds[0]+bounds[1]]
        test_data = data_batch[bounds[0]+bounds[1]:]
        return np.array(train_data), np.array(dev_data), np.array(test_data)

    def init_model_optimizer(self):
        model = GeneratingLSTM(self.word2idx, wordvec_dim=self.args['model']['wordvec_dim'], hidden_dim=self.args['model']['hidden_dim'])
        optimizer = optim.__dict__[self.args['model']['optimizer']](params=model.params, config=self.args['model']['optim_config'])
        return model, optimizer
        
    def save_model(self):
        path = 'ckpt/'
        if not os.path.exists(path):
            os.mkdir(path)
        np.savez_compressed(path+'model',
                            params=self.model.params,
                            optim_config=self.optimizer.config, 
                            epoch=self.epoch,
                            epoch_loss=self.epoch_loss,
                            epoch_perp=self.epoch_perp)
        print('Model Saved.')

    def load_model(self):
        path = 'ckpt/'
        if os.path.exists(path+'model.npz'):
            ckpt = dict(np.load(path+'model1.npz'))
            self.model.params = ckpt['params'].item()
            self.optimizer = optim.__dict__[self.args['model']['optimizer']](params=self.model.params, config=self.args['model']['optim_config'])
            #self.optimizer.config = ckpt['optim_config']
            self.epoch = ckpt['epoch'].item()
            #self.epoch_loss = ckpt['epoch_loss']
            #self.epoch_perp = ckpt['epoch_perp']
            print('Model Loaded.')
        else:
            print('No Saved Model!')
    
    def train(self):
        self.epoch += 1
        loss_record = []
        grad_record = []
        perp_record = []
        begin_time = time.time()
        
        widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar('#'),
                   ' ', progressbar.Timer(), ' ', progressbar.ETA()]#, ' ', progressbar.FileTransferSpeed()]
        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(self.train_data)).start()
        
        for (i, input_data) in enumerate(self.train_data):
            progress.update(i + 1)
            sentence_in = input_data[:, :-1] # [bs, 124]
            sentence_out = input_data[:, 1:]
            
            result, _ = self.model.forward(sentence_in)
            
            mask = (sentence_out != self.word2idx['<OOV>'])
            loss, dout = self.metric(result, sentence_out, mask)
            
            grads = self.model.backward(dout)
            
            self.optimizer.step(grads)
            
            perp = np.mean(self.calculate_perplexity())
            
            loss_record.append(loss)
            perp_record.append(perp)
            
            if perp < self.args['perplexity_threshold']:
                print(':-) perplexity!')
                break
        self.epoch_loss.append(np.mean(loss_record))
        self.epoch_perp.append(np.mean(perp_record))
        progress.finish()
        utility.clear_progressbar()
        
        print('--------------------------------------------------')
        print('The epoch :', self.epoch)
        print('The epoch costs time: %.2f' % (time.time()-begin_time), 's')
        print('The training loss is: %f' % (np.mean(loss_record)))
        print('The perplexity is: %f' % (np.mean(perp_record)))

    def test(self):
        loss_record = []
        begin_time = time.time()
        
        widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar('#'),
                   ' ', progressbar.Timer(), ' ', progressbar.ETA()]#, ' ', progressbar.FileTransferSpeed()]
        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(self.train_data)).start()
        
        for (i, input_data) in enumerate(self.test_data):
            progress.update(i + 1)
            sentence_in = input_data[:, :-1]
            sentence_out = input_data[:, 1:]
            result, _ = self.model.forward(sentence_in)
            mask = (sentence_out != self.word2idx['<OOV>'])
            loss, dout = self.metric(result, sentence_out, mask)
            loss_record.append(loss)
        self.test_loss = np.mean(loss_record)
        progress.finish()
        utility.clear_progressbar()
        print('--------------------------------------------------')
        print('The test procedure costs time: %.2f' % (time.time()-begin_time), 's')
        print('The test loss is: %f' % (self.test_loss))

    def generate_mini(self, start_word, num_sent=8, sent_len=5, max_k=6):
        result = [start_word]
        input = np.array([self.word2idx.get(start_word, self.word2idx['<OOV>'])]).reshape(1, 1)
        hidden = None
        i = 1
        while True:
            output, hidden  = self.model.forward(input, hidden)
            output_flat = list(np.squeeze(output))
            output_flat_sorted = sorted(output_flat)[-max_k:]
            idx_list = [output_flat.index(out) for out in output_flat_sorted]
            #max_idx = np.argmax(output, axis=-1)[0][0]
            max_idx = max(int(idx_list[int(np.random.uniform(0, len(idx_list)))]), 0)
            w = self.idx2word[max_idx]
            if i > 0 and i % sent_len == 0:
                result.append('，')
            if i > 0 and i % (2 * sent_len) == 0:
                result[-1] = '。'
            if w == '<EOS>' or i > num_sent * sent_len:
                break
            #if i > num_sent * sent_len:
            #    break
            #if w == '<EOS>' or w == '<START>':
            #    continue
            result.append(w)
            input = np.array([max_idx]).reshape(1, 1)
            i += 1
        poem = ''
        for i in range(len(result)):
            poem += str(result[i])
            if result[i] == '。':
                print(poem)
                poem = ''

    def generate_big(self, start_word):
        result = [start_word]
        input = np.array([self.word2idx[start_word]]).reshape(1, 1)
        hidden = None
        for i in range(self.args['max_length']):
            output, hidden  = self.model.forward(input, hidden)
            max_idx = np.argmax(output, axis=-1)[0][0]
            w = self.idx2word[max_idx]
            result.append(w)
            input = np.array([max_idx]).reshape(1, 1)
            if w == '<EOS>':
                break
        print(result)

    def calculate_perplexity(self):
        perp = 0
        self.dev_data = self.dev_data[:5]
        for (i, input_data) in enumerate(self.dev_data):
            sentence_in = input_data[:, :-1]
            sentence_out = input_data[:, 1:]
            result, _ = self.model.forward(sentence_in)
            perp += utility.perplexity(result)
        return perp / (i+1)

    def plot(self):
        print(self.epoch_loss)
        nums = len(self.epoch_loss)
        x = range(nums)
        plt.suptitle('loss and perplexity curves in training procedure', fontsize=16)
        plt.title('optimizer: %s, learning rate: %f, test loss: %f' %
                 (self.args['model']['optimizer'],
                  self.args['model']['optim_config']['learning_rate'],
                  self.test_loss),
                  fontsize=10)
        plt.plot(x, self.epoch_loss, label='loss', color='#FFA500')
        plt.plot(x, self.epoch_perp, label='perplexity', color='cyan')
        plt.legend()
        plt.show()

