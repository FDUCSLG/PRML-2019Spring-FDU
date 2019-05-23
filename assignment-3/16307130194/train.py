from fastNLP import DataSet
from fastNLP import Trainer
from fastNLP.core.optimizer import Adam

from dataset import get_dataset
from config import Config
from model import PoetryModel
from utils import Perplexity


def train():
    config = Config()

    train_data, dev_data, vocabulary = get_dataset(config.data_path)
    print('Train data size:', len(train_data))
    print('Dev data size:', len(dev_data))
    print('Vocab size:', len(vocabulary))

    poetry_model = PoetryModel(vocabulary_size=len(vocabulary), embedding_size=config.embedding_size,
                               hidden_size=config.hidden_size)
    perplexity = Perplexity()
    if config.optimizer == 'adam':
        optimizer = Adam(lr=config.lr , weight_decay=config.weight_decay)

    trainer = Trainer(train_data=train_data, model=poetry_model, loss=None,
                      metrics=perplexity, n_epochs=config.epoch, batch_size=config.batch_size,
                      print_every=config.print_every, validate_every=config.validate_every,
                      dev_data=dev_data, save_path=config.save_path, 
                      optimizer=optimizer, check_code_level=)


def main(**kwargs):
    for k, v in kwargs.items():
        setattr(config, k, v)

    config.device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    device = config.device
    vis = Visualizer(env=config.env)

    # 获取数据
    data, word2ix, ix2word = get_data(config)
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    configimizer = torch.configim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    for epoch in range(config.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            configimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            configimizer.step()

            loss_meter.add(loss.item())

            # 可视化
            if (1 + ii) % config.plot_every == 0:

                if os.path.exists(config.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 诗歌原文
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                           for _iii in range(data_.shape[1])][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')

        torch.save(model.state_dict(), '%s_%s.pth' % (config.model_prefix, epoch))


if __name__ == "__main__":
    train()
