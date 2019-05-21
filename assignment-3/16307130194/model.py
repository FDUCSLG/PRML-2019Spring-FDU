import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self):
        super(PoetryModel, self).__init__()

        self.hidden