import numpy as np
import sys
import argparse
import pickle
import math
sys.path.append('../')

import torch.nn as nn
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torchnet import meter
from fastNLP import Trainer
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core import optimizer as fastnlp_optim
from fastNLP.core.callback import EarlyStopCallback
from fastNLP import Batch
from copy import deepcopy

from MyUtils.dataset import Config,PoemData
from MyUtils.MyLSTM import MyLSTM
from MyUtils.losses import MyCrossEntropyLoss
from MyUtils.metrics import MyPerplexityMetric
from model import MyPoetryModel, PoetryModel,FastNLPPoetryModel
from generater import generate_poet

parser = argparse.ArgumentParser()

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--train_torch_lstm', action='store_true',
                        help='train the model')
    parser.add_argument('--train_fastnlp', action='store_true',
                        help='train the model')
    parser.add_argument('--generate', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--contrain', action='store_true',
                        help='reload the model and continue to train model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout', type=float, default=0,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_seq_len', type=int, default=50,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_gen_len', type=int, default=50,
                                help='max length of passage')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def train(conf,args=None):
    pdata = PoemData()
    pdata.read_data(conf)
    pdata.get_vocab()
    model = MyPoetryModel(pdata.vocab_size,conf.embedding_dim,conf.hidden_dim)

    train_data = pdata.train_data
    test_data = pdata.test_data

    train_data = torch.from_numpy(np.array(train_data['pad_words']))
    dev_data = torch.from_numpy(np.array(test_data['pad_words']))

    dataloader = DataLoader(train_data, batch_size=conf.batch_size,shuffle=True,num_workers=conf.num_workers)
    devloader = DataLoader(dev_data,batch_size=conf.batch_size,shuffle=False,num_workers=conf.num_workers)
    
    optimizer = Adam(model.parameters(),lr = conf.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()


    if conf.load_best_model:
        model.load_state_dict(torch.load(conf.beat_model_path))
    if conf.use_gpu:
        model.cuda()
        criterion.cuda()
    step=0
    bestppl = 1e9
    early_stop_controller=0
    for epoch in range(conf.n_epochs):
        losses=[]
        loss_meter.reset()
        model.train()
        for i,data in enumerate(dataloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            optimizer.zero_grad()
            output, _= model(input)
            loss = criterion(output, target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loss_meter.add(loss.item())
            step+=1
            if step%100==0:
                print("epoch_%d_step_%d_loss:%0.4f" % (epoch+1, step,loss.item()))
        train_loss = float(loss_meter.value()[0])
        
        model.eval()
        for i,data in enumerate(devloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            output, _= model(input)
            loss = criterion(output, target.view(-1))
            loss_meter.add(loss.item())
        ppl = math.exp(loss_meter.value()[0])
        print("epoch_%d_loss:%0.4f , ppl:%0.4f" % (epoch+1,train_loss,ppl) )

        if epoch % conf.save_every == 0:
            torch.save(model.state_dict(),"{0}_{1}".format(conf.model_prefix,epoch))

            fout = open("{0}out_{1}".format(conf.out_path,epoch),'w',encoding='utf-8')
            for word in list('日红山夜湖海月'):
                gen_poetry = generate_poet(model, word, pdata.vocab, conf)
                fout.write("".join(gen_poetry)+'\n\n')
            fout.close()
        if ppl<bestppl:
            bestppl = ppl
            early_stop_controller = 0
            torch.save(model.state_dict(),"{0}_{1}".format(conf.best_model_path,"best_model"))
        else:
            early_stop_controller += 1
        if early_stop_controller>10:
            print("early stop.")
            break

def train_torch_lstm(conf,args=None):
    pdata = PoemData()
    pdata.read_data(conf)
    pdata.get_vocab()
    if conf.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = PoetryModel(pdata.vocab_size,conf,device)

    train_data = pdata.train_data
    test_data = pdata.test_data

    train_data = torch.from_numpy(np.array(train_data['pad_words']))
    dev_data = torch.from_numpy(np.array(test_data['pad_words']))

    dataloader = DataLoader(train_data, batch_size=conf.batch_size,shuffle=True,num_workers=conf.num_workers)
    devloader = DataLoader(dev_data,batch_size=conf.batch_size,shuffle=True,num_workers=conf.num_workers)
    
    optimizer = Adam(model.parameters(),lr = conf.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()


    if conf.load_best_model:
        model.load_state_dict(torch.load(conf.beat_model_path))
    if conf.use_gpu:
        model.cuda()
        criterion.cuda()
    step=0
    bestppl = 1e9
    early_stop_controller=0
    for epoch in range(conf.n_epochs):
        losses=[]
        loss_meter.reset()
        model.train()
        for i,data in enumerate(dataloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            optimizer.zero_grad()
            output, _= model(input)
            loss = criterion(output, target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loss_meter.add(loss.item())
            step+=1
            if step%100==0:
                print("epoch_%d_step_%d_loss:%0.4f" % (epoch+1, step,loss.item()))
        train_loss = float(loss_meter.value()[0])
        
        model.eval()
        for i,data in enumerate(devloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            output, _= model(input)
            loss = criterion(output, target.view(-1))
            loss_meter.add(loss.item())
        ppl = math.exp(loss_meter.value()[0])
        print("epoch_%d_loss:%0.4f , ppl:%0.4f" % (epoch+1,train_loss,ppl) )

        if epoch % conf.save_every == 0:
            torch.save(model.state_dict(),"{0}_{1}".format(conf.model_prefix,epoch))

            fout = open("{0}out_{1}".format(conf.out_path,epoch),'w',encoding='utf-8')
            for word in list('日红山夜湖海月'):
                gen_poetry = generate_poet(model, word, pdata.vocab, conf)
                fout.write("".join(gen_poetry)+'\n\n')
            fout.close()
        if ppl<bestppl:
            bestppl = ppl
            early_stop_controller = 0
            torch.save(model.state_dict(),"{0}".format(conf.best_model_path))
        else:
            early_stop_controller += 1
        if early_stop_controller>conf.patience:
            print("early stop.")
            break

def train_fastnlp(conf,args=None):
    pdata = PoemData()
    pdata.read_data(conf)
    pdata.get_vocab()
    if conf.use_gpu:
        device = torch.device('cuda')
    else:
        device = None
    conf.device = device
    model = FastNLPPoetryModel(pdata.vocab_size,conf.embedding_dim,conf.hidden_dim,device)
    train_data = pdata.train_data
    test_data = pdata.test_data
    train_data.apply(lambda x: x['pad_words'][:-1], new_field_name="input")
    train_data.apply(lambda x: x['pad_words'][1:], new_field_name="target")
    test_data.apply(lambda x: x['pad_words'][:-1], new_field_name="input")
    test_data.apply(lambda x: x['pad_words'][1:], new_field_name="target")
    train_data.set_input("input")
    train_data.set_target("target")
    test_data.set_input("input")
    test_data.set_target("target")

    loss = MyCrossEntropyLoss(pred='output',target='target')
    metric = MyPerplexityMetric(pred='output',target='target')
    optimizer = fastnlp_optim.Adam(lr=conf.learning_rate, weight_decay=0)
    overfit_model = deepcopy(model)
    overfit_trainer = Trainer(model=overfit_model,device= conf.device,
                            batch_size=conf.batch_size,n_epochs=conf.n_epochs,
                            train_data=train_data,dev_data=test_data,
                            loss=loss,metrics=metric,optimizer=optimizer,
                            save_path=conf.best_model_path,
                            validate_every=conf.save_every,
                            metric_key="-PPL",
                            callbacks=[EarlyStopCallback(conf.patience)]
                            )
    print(overfit_trainer.train())

def prepare(conf,args):
    pass

def generate(conf,args):
    pass

def contrain(conf,args):

    pass

def run():
    conf = Config()
    args = parse_args()
    args.train = True

    if args.prepare:
        prepare(conf,args)
    if args.train:
        train(conf,args)
    if args.train_torch_lstm:
        train_torch_lstm(conf,args)
    if args.train_fastnlp:
        train_fastnlp(conf,args)
    if args.generate:
        generate(conf,args)
    if args.contrain:
        contrain(conf,args)

if __name__ == "__main__":
    run()