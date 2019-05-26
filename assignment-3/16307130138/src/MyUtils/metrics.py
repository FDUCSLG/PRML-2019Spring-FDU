    
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.core.metrics import MetricBase

class MyPerplexityMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super(MyPerplexityMetric, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.count = 0
        self.PP_sum = 0
    
    def evaluate(self, pred, target):
        batch, seq_len = target.shape
        pred = pred.float()
        target = target.view(-1,1).long()
        pred = F.softmax(pred, dim=1)

        target_onehot = torch.zeros_like(pred)
        target_onehot = target_onehot.scatter(1, target, 1)

        x = torch.sum(torch.mul(pred, target_onehot), dim=1).view(batch, seq_len)
        self.count += batch
        self.PP_sum += torch.sum(torch.exp(torch.mean(-torch.log(x), dim=1)))
 
    def get_metric(self, reset=True):
        evaluate_result = {'PPL':self.PP_sum/self.count}
        if reset:
            self.count = 0
            self.PP_sum = 0
        return evaluate_result