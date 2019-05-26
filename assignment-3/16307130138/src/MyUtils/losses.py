import torch
import torch.nn.functional as F
from fastNLP.core.losses import LossBase

class MyCrossEntropyLoss(LossBase):
    def __init__(self, pred=None, target=None, padding_idx=-100):
        super(MyCrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = padding_idx

    def get_loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target.view(-1),
                               ignore_index=self.padding_idx)