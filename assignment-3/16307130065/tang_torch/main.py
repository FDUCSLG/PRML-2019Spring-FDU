from torch.utils.data import DataLoader
import torch.optim as opt
from data import *
from model import *
from generate import *


def train():
    device = torch.device("cuda")
    vocab, train_data = pre_process()
    vocab_size = len(vocab)
    train_data = torch.from_numpy(train_data)
    data_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    model = PoetryModel(vocab_size, Config.embedding_dim, Config.hidden_dim)
    model.to(device)
    optimizer = opt.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(Config.epoch):
        count = 0
        for i, data in enumerate(data_loader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            optimizer.zero_grad()
            input_, target = data[:-1, :], data[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            count += 1
            perplexity = torch.mean(torch.exp(loss)).item()
            print("epoch:%d step:%d loss:%0.8f perplexity:%0.8f" % (epoch+1, count, loss.item(), perplexity))

    gen_poetry = ''.join(generate(model, '日', vocab))
    print(gen_poetry)
    torch.save(model.state_dict(), 'model/poem.pth')


if __name__ == '__main__':
    train()
    # device = torch.device("cuda")
    # vocab, train_data = pre_process()
    # vocab_size = len(vocab)
    # train_data = torch.from_numpy(train_data)
    # data_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    # model = PoetryModel(vocab_size, Config.embedding_dim, Config.hidden_dim)
    # model.to(device)
    # model.load_state_dict(torch.load('model/poem.pth', 'cuda'))
    # gen_poetry = ''.join(generate(model, '海', vocab))
    # print(gen_poetry)
    # print(len(gen_poetry))
