from fastNLP.core.losses import LossBase
from fastNLP import Callback
from fastNLP.core.metrics import MetricBase
from fastNLP.core.optimizer import Optimizer

import torch
import torch.nn.functional as F
import time


start_time = time.time()


class TimingCallback(Callback):
    def on_epoch_end(self):
        print('Sum Time: {:d}ms\n\n'.format(round((time.time() - start_time) * 1000)))


class Perplexity(MetricBase):
    def __init__(self, pred=None, target=None):
        super(Perplexity, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.ppl_sum = 0.
        self.target_count = 0.

    def evaluate(self, pred, target):
        batch, seq_len = target.shape
        pred = pred.float()
        target = target.view(-1,1).long()
        pred = F.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot = target_onehot.scatter(1, target, 1)
        x = torch.sum(torch.mul(pred, target_onehot), dim=1).view(batch, seq_len)
        self.ppl_sum += torch.sum(torch.exp(torch.mean(-torch.log(x), dim=1)))
        self.target_count += batch

    def get_metric(self, reset=True):
        metric = {
            'PPL': self.ppl_sum / self.target_count
        }
        if reset:
            self.ppl_sum = 0.
            self.target_count = 0.
        return metric


class Loss(LossBase):
    def __init__(self, pred=None, target=None, padding_idx=-1):
        super(Loss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = padding_idx

    def get_loss(self, pred, target):
        loss = F.cross_entropy(input=pred, target=target.view(-1), ignore_index=self.padding_idx)
        print(loss)
        return loss


class Adagrad(Optimizer):
    def __init__(self, lr=1e-3, lr_decay=0, weight_decay=0, initial_accumulator_value=0, model_params=None):
        super(Adagrad, self).__init__(model_params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            return torch.optim.Adagrad(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adagrad(self._get_require_grads_param(self.model_params), **self.settings)


class Adadelta(Optimizer):
    def __init__(self, lr=1e-3, rho=0.9, eps=1e-6, weight_decay=0, model_params=None):
        super(Adadelta, self).__init__(model_params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            return torch.optim.Adadelta(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adadelta(self._get_require_grads_param(self.model_params), **self.settings)

