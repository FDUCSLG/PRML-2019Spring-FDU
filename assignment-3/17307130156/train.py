import os, glob
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

from model import RecurrentNetwork as CharRNN
from utils import DataProvider


# directory = './raw_data'
checkpoint_path = './checkpoints/poetry_gen_test.pt'
# pp_path = 'pp.log'
# loss_path = 'loss.log'

# Hyperparameters
learning_rate = 0.01
embedding_dim = 256
hidden_size = 128
num_layers = 2
num_epochs = 10000
train_size = 0.8
batch_size = 128
save_per_num_steps = 20


included_extensions = ['.json']
# files = [directory + '/' + fn for fn in os.listdir(directory) if any(fn.endswith(ext) for ext in included_extensions)]
files = ['raw_data/out.json']
random.shuffle(files)

print ('Preparing data...')
provider = DataProvider(files, batch_size=batch_size, padding_value=0)

vocab = provider.vocab
vocab_size = len(vocab)

print ('Vocab size: ', vocab_size)


model = CharRNN(vocab_size=vocab_size,
                target_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers)
criterion = nn.NLLLoss(ignore_index=vocab.padding_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

e = 0
try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch']

    model.eval()
    print ('Load previous model and optimizer!')
except:
    print ('No saved model found!')


def train(epochs):
    cnt = 0
    pre_pp = -1
    for epoch in range(epochs):

        print ('Epoch: {}'.format(epoch))

        for input_batch, target_batch, sorted_lengths in provider.padded_batches():
    
            input_batch = Variable(input_batch)
            target_batch = Variable(target_batch)

            model.zero_grad()
            model.init_hidden(input_batch.shape[1]) # batch size

            output_batch, _ = model(input_batch, sorted_lengths) 

            loss = model.loss(output_batch, target_batch, vocab.padding_idx)
            loss.backward()
            optimizer.step()
            print (f'Loss: {loss}')
            '''
            # Saving loss used for plotting
            with open(loss_path, 'a') as f:
                f.write(f'{loss}\n')
            '''

            cnt += 1
            if cnt == save_per_num_steps:

                cnt = 0

                torch.save({
                    'epoch': e + epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print ('Save model: total epochs ', e + epoch)

        pp = 0
        num_samples = 0
        for input_batch, target_batch, sorted_lengths in provider.devset_batches():
            input_batch = Variable(input_batch)
            target_batch = Variable(target_batch)

            batch_size = input_batch.shape[1]
            
            state = model.init_hidden(batch_size)
            output_batch, _ = model(input_batch, sorted_lengths) 

            pp += model.perplexity(output_batch, target_batch, vocab.padding_idx).item()
            num_samples += batch_size

        pp = pp / num_samples
        print (f'Average Perplexity: {pp}')
        if pre_pp > 0 and pp > pre_pp:
            print ('Training terminates !')
            return 
        pre_pp = pp
        '''
        # Saving perplexity used for plotting
        with open(pp_path, 'a') as f:
            f.write(f'{pp}\n')
        '''
        
        
if __name__ == '__main__':

    train(num_epochs)







