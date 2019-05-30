import numpy as np
import torch

def find_class_by_name(name, modules):
    """ searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def load_model(model, model_path):
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    return True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)  

def dtanh(y):
    return 1 - y * y
