import sys, os
import time
import numpy as np
import torch
import torch.utils.data as D
from dataUtils import get_data
from model import PoetryModel
from torch import nn


class Config(object):
    use_gpu = True
    batch_size = 128
    lr = 1e-3
    epoch = 50
    # gen_every = 25
    gen_every = 50
    max_gen_len = 200
    save_every = 5
    pp_every = 100
    tolerance = 10
    filepath = "tang.npz"
    pass

config = Config()
use_cuda = config.use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
print(device)


def check_perplexity(model, dev_dataloader):
    criterion = nn.CrossEntropyLoss()
    perplexity = 0
    with torch.no_grad():
        for i, data_ in enumerate(dev_dataloader):
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            perplexity += torch.exp(loss)
    perplexity /= i
    print(perplexity)
    return perplexity

def generate(model, start_words, vocab):
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([vocab['<START>']]).view(1, 1).long()
    if use_cuda: input = input.cuda()
    hidden = None

    for i in range(config.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([vocab[w]]).view(1, 1)
        else:
            # top_index = output.data[0].topk(1)[1][0].item()
            soft_output = nn.functional.softmax(output.data[0])
            index = torch.multinomial(soft_output, 1).cpu().numpy()[0]
            w = vocab.to_word(index)
            results.append(w)
            input = input.data.new([index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

def train(**kwargs):
    for k, v in kwargs.items():
        setattr(config, k, v)

    device=torch.device('cuda') if use_cuda else torch.device('cpu')

    # 获取数据
    data, vocab = get_data(config.filepath)
    np.random.shuffle(data)
    l = len(data)
    dev_data = data[:l//5-1]
    data = data[l//5:]
    data = torch.from_numpy(data)
    dev_data = torch.from_numpy(dev_data)
    dataloader = D.DataLoader(data, batch_size=config.batch_size,
                            shuffle=True, num_workers=4)
    dev_dataloader = D.DataLoader(dev_data, batch_size=config.batch_size,
                            shuffle=True, num_workers=4)

    # 模型定义
    model = PoetryModel(len(vocab.word2idx), 128, 256)

    # if config.model_path:
    #     model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    # SGD, SGD with momentum, Nesterov, Adagrad, Adadelta, Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.Adadelta(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    criterion = nn.CrossEntropyLoss()

    

    pre_pp = 0
    cnt = -1
    loss_his = []
    pp_his = []
    for epoch in range(config.epoch):
        for ii, data_ in enumerate(dataloader):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "step", ii, "loss", loss.item())
            loss_his.append(loss.item())

            # 测试
            if (1 + ii) % config.gen_every == 0:
                # "'春江花月夜凉如水'"
                word = "春"
                gen_poetry = ''.join(generate(model, word, vocab))
                print(gen_poetry)
            

            if (1 + ii) % config.pp_every == 0:
                pp = check_perplexity(model, dev_dataloader)
                if pre_pp < pp:
                    cnt += 1
                pre_pp = pp
                print(pp.cpu().numpy())
                pp_his.append(pp.cpu().numpy())

                if cnt >= config.tolerance:
                    torch.save(model.state_dict(), '%s_final.pth' % str(int(time.time())))
                    print("epoch", epoch, "step", ii, "final loss", loss.item())
                    for word in ["日","红","山","夜","湖","海","月"]:
                        gen_poetry = ''.join(generate(model, word, vocab))
                        print(gen_poetry)
                    return loss_his, pp_his
        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.epoch:
            torch.save(model.state_dict(), '%s_%s.pth' % (str(int(time.time())), str(epoch)))
    return loss_his, pp_his
if __name__ == "__main__":
    loss_his, pp_his = train()
    np.save("loss_gen", loss_his)
    np.save("pp_gen", pp_his)