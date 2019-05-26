# -*- coding: utf-8 -*-
import os
os.sys.path.append('..')
import math
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester
from fastNLP import CrossEntropyLoss
from fastNLP import NLLLoss
from fastNLP import Adam
from fastNLP import SGD
from fastNLP import AccuracyMetric
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import string
import random
from lstm import *

hanzi_punctuation = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
start_token = 'S'
end_token = 'E'
# 超参数
vocab_size = 5416
sentence_len = 49
batch_size = 64
input_size = 512
lstm_hidden_size = 512
learning_rate = 1e-3
temperature = 0.8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("vocab_size:", vocab_size)
print("sentence_len:", sentence_len)
print("batch_size:", batch_size)
print("input_size:", input_size)
print("lstm_hidden_size:", lstm_hidden_size)
print("learning_rate:", learning_rate)
print("temperature:", temperature)


def read_vocab(file_name):
    # 读入vocab文件
    with open('vocab.txt') as f:
        lines = f.readlines()
    vocabs = []
    for line in lines:
        vocabs += list(line.strip())

    # 实例化Vocabulary
    vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    # 将vocabs列表加入Vocabulary
    vocab.add_word_lst(vocabs)
    # 构建词表
    vocab.build_vocab()
    return vocab


def gen_poem(model, vocab, begin_word):
    poem = begin_word
    # input = torch.ones([1], dtype=torch.int32)
    # end_input = torch.ones([1], dtype=torch.int32)
    input = torch.tensor([vocab[begin_word]], device=device)
    end_input = torch.tensor([vocab[end_token]], device=device)
    predict = model.generate_seq(input, end_input, temperature=temperature)
    for pred in predict:
        word = vocab.to_word(pred)
        poem += word
    return poem


train_data = DataSet().load("train_data.txt")
dev_data = DataSet().load("dev_data.txt")
vocab = read_vocab("vocab.txt")
# for i in range(len(vocab)):
#     print(i, vocab.to_word(i))

lstm_model = RNN_model(vocab_len=len(vocab), embedding_size=input_size, lstm_hidden_size=lstm_hidden_size)


def perplexity(output, target):
    with torch.no_grad():
        output = output.permute(0, 2, 1)
        batch_size, seq_len = output.shape[0], output.shape[1]
        perp = 0
        for x, y in zip(output, target):
            temp = 0
            for xi, yi in zip(x, y):
                temp -= math.log(xi[yi])
            perp += math.exp(temp / seq_len)
    
    return perp / batch_size


def validation(batch_size, batch_iterator):
    with torch.no_grad():
        perp_sum = 0
        count = 0
        # loss_sum = 0
        # loss_calc = nn.CrossEntropyLoss(reduction='mean')
        softmax = nn.Softmax(dim=1)
        for batch_x, batch_y in batch_iterator:
            x = batch_x['sentence'].cuda()
            y = batch_y['target'].cuda()
            output = softmax(lstm_model(x)['pred'])
            # print(sum(output[0][i][0] for i in range(output.shape[1])))
            # loss_sum += loss_calc(output, y) / output.shape[0]
            perp = perplexity(output, y)
            perp_sum += perp
            count += 1
    return perp_sum / count #, loss_sum / count


def my_trainer(epochs, batch_size, lr, model_name, optimizer):
    lstm_model.to(device)
    loss_calc = nn.CrossEntropyLoss(reduction='mean')
    batch_iterator = Batch(dataset=train_data, batch_size=batch_size, sampler=RandomSampler())
    batch_iterator2 = Batch(dataset=dev_data, batch_size=batch_size, sampler=RandomSampler())
    loss_list = []
    metric_list = []
    # vali_loss_list = []
    count = 0
    min_perp = 0
    min_perp_epoch = 0
    for epo in range(epochs):
        for batch_x, batch_y in batch_iterator:
            x = batch_x['sentence'].cuda()
            y = batch_y['target'].cuda()
            optimizer.zero_grad()
            output = lstm_model(x)['pred']
            # seq_len = output.shape[2]
            loss = loss_calc(output, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_list.append(loss.item())
                if count % 10 == 0:
                    print("step:", count, ", loss =", loss.item())
            count += 1
        
        perp = validation(batch_size, batch_iterator2)
        # vali_loss_list.append(vali_loss)
        if epo == 0 or min_perp >= perp:
            min_perp = perp
            # torch.save(lstm_model.state_dict(), model_name)
            min_perp_epoch = epo + 1
        with torch.no_grad():
            metric_list.append(perp)
            print("epochs =", epo + 1, ", perplexity =", perp)
            # print(gen_poem(lstm_model, vocab, "日"))
            # print(gen_poem(lstm_model, vocab, "红"))
            # print(gen_poem(lstm_model, vocab, "山"))
            # print(gen_poem(lstm_model, vocab, "夜"))
            # print(gen_poem(lstm_model, vocab, "湖"))
            # print(gen_poem(lstm_model, vocab, "海"))
            # print(gen_poem(lstm_model, vocab, "月"))
    
    print("finish train, best model in epoch", min_perp_epoch, ", perplexity =", min_perp)
    # torch.save(lstm_model.state_dict(), model_name+"_final")

    plt.plot(range(1, len(loss_list)+1), loss_list, label='train_loss')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.title('Adam\nlearning_rate=%.1e, betas=(0.5, 0.99)' % (lr))
    plt.legend()
    plt.show()
    plt.plot(range(1, len(metric_list)+1), metric_list, label='perplexity')
    plt.xlabel('epochs')
    plt.ylabel('Perplexity')
    plt.title('Adam\nlearning_rate=%.1e, betas=(0.5, 0.99)' % (lr))
    plt.legend()
    plt.show()
    return loss_list


# trainer = Trainer(
#     train_data=train_data,
#     model=lstm_model,
#     loss=CrossEntropyLoss(pred='pred', target='target'),
#     # metrics=AccuracyMetric(),
#     n_epochs=50,
#     batch_size=batch_size,
#     print_every=10,
#     # validate_every=-1,
#     # dev_data=dev_data,
#     use_cuda=True,
#     optimizer=Adam(lr=learning_rate, betas=(0.5, 0.99)),
#     check_code_level=2,
#     # metric_key='acc',
#     use_tqdm=False,
# )

model_name = "./huge_poem_gen_vc%d_input%d_lr%.0e_batch%d_hid%d" % (vocab_size, input_size, learning_rate, batch_size, lstm_hidden_size) 
# lstm_model.load_state_dict(torch.load(model_name))

# optimizer = torch.optim.RMSprop(lstm_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, betas=(0.5, 0.99))
loss_list1 = my_trainer(epochs=5, batch_size=batch_size, 
        lr=learning_rate, model_name=model_name, optimizer=optimizer)
plt.plot(range(1, len(loss_list1)+1), loss_list1, label='Adam', alpha=0.8)

lstm_model = RNN_model(vocab_len=len(vocab), embedding_size=input_size, lstm_hidden_size=lstm_hidden_size)
optimizer = torch.optim.RMSprop(lstm_model.parameters())
loss_list2 = my_trainer(epochs=5, batch_size=batch_size, 
        lr=learning_rate, model_name=model_name, optimizer=optimizer)
plt.plot(range(1, len(loss_list2)+1), loss_list2, label='RMSprop', alpha=0.8)

lstm_model = RNN_model(vocab_len=len(vocab), embedding_size=input_size, lstm_hidden_size=lstm_hidden_size)
optimizer = torch.optim.Adagrad(lstm_model.parameters())
loss_list3 = my_trainer(epochs=5, batch_size=batch_size, 
        lr=learning_rate, model_name=model_name, optimizer=optimizer)
plt.plot(range(1, len(loss_list3)+1), loss_list3, label='Adagrad', alpha=0.8)

lstm_model = RNN_model(vocab_len=len(vocab), embedding_size=input_size, lstm_hidden_size=lstm_hidden_size)
optimizer = torch.optim.SGD(lstm_model.parameters(), lr=1e-3)
loss_list4 = my_trainer(epochs=5, batch_size=batch_size, 
        lr=learning_rate, model_name=model_name, optimizer=optimizer)
plt.plot(range(1, len(loss_list4)+1), loss_list4, label='SGD', alpha=0.7)

lstm_model = RNN_model(vocab_len=len(vocab), embedding_size=input_size, lstm_hidden_size=lstm_hidden_size)
optimizer = torch.optim.SGD(lstm_model.parameters(), lr=1e-1, momentum=0.9)
loss_list5 = my_trainer(epochs=5, batch_size=batch_size, 
        lr=learning_rate, model_name=model_name, optimizer=optimizer)
plt.plot(range(1, len(loss_list5)+1), loss_list5, label='SGD_momentum', alpha=0.7)


plt.xlabel('steps')
plt.ylabel('Loss')
plt.title('train_loss')
plt.legend()
plt.show()

# trainer.train()

torch.save(lstm_model.state_dict(), model_name)
print("finish save model")

print(gen_poem(lstm_model, vocab, "日"))
print(gen_poem(lstm_model, vocab, "红"))
print(gen_poem(lstm_model, vocab, "山"))
print(gen_poem(lstm_model, vocab, "夜"))
print(gen_poem(lstm_model, vocab, "湖"))
print(gen_poem(lstm_model, vocab, "海"))
print(gen_poem(lstm_model, vocab, "月"))
