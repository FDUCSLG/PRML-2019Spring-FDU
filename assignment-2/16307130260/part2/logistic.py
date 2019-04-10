import numpy as np

from layer_utils import *

class logistic_model:
    def __init__(self, input_dims, class_nums, weight_scale=1e-3, reg=0):
        self.params = {}
        self.reg = reg
        rng = np.random.RandomState(2333)
        self.params["w"] = rng.normal(0.0, weight_scale, (input_dims, class_nums))
        self.params["b"] = np.zeros(class_nums)

    def loss(self, x, y=None):
        fc_out, fc_cache = affine_forward(self.params["w"], x, self.params["b"])
        grads = {}
        # for test, just return the output of linear layer for prodict
        if y is None:
            return fc_out, grads
        softmax_out, softmax_dout = softmax_loss(fc_out, y)
        
        grads["w"], grads["b"] = affine_backward(softmax_dout, fc_cache)

        # add reg term
        loss = softmax_out + self.reg * 0.5 * np.sum(self.params["w"] ** 2)
        grads["w"] += self.reg * self.params["w"]
        return loss, grads

