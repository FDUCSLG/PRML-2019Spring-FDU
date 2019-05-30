from fastNLP import Trainer
from fastNLP import Adam, SGD
from fastNLP import RandomSampler
from fastNLP.core.callback import EarlyStopCallback

from dataset import get_dataset
from config import Config
from model import PoetryModel
from utils import Perplexity
from utils import Loss
from utils import TimingCallback
from utils import Adagrad
from utils import Adadelta


def train():
    config = Config()

    train_data, dev_data, vocabulary = get_dataset(config.data_path)

    poetry_model = PoetryModel(vocabulary_size=len(vocabulary), embedding_size=config.embedding_size,
                               hidden_size=config.hidden_size)
    loss = Loss(pred='output', target='target')
    perplexity = Perplexity(pred='output', target='target')

    print("optimizer:", config.optimizer)
    print("momentum:", config.momentum)
    if config.optimizer == 'adam':
        optimizer = Adam(lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = SGD(lr=config.lr, momentum=config.momentum)
    elif config.optimizer == 'adagrad':
        optimizer = Adagrad(lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adadelta':
        optimizer = Adadelta(lr=config.lr, rho=config.rho, eps=config.eps, weight_decay=config.weight_decay)

    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)

    trainer = Trainer(train_data=train_data, model=poetry_model, loss=loss,
                      metrics=perplexity, n_epochs=config.epoch, batch_size=config.batch_size,
                      print_every=config.print_every, validate_every=config.validate_every,
                      dev_data=dev_data, save_path=config.save_path,
                      optimizer=optimizer, check_code_level=config.check_code_level,
                      metric_key="-PPL", sampler=RandomSampler(), prefetch=False,
                      use_tqdm=True, device=config.device, callbacks=[timing, early_stop])
    trainer.train()


if __name__ == "__main__":
    train()
