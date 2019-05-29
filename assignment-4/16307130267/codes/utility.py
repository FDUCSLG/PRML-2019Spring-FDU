import torch
import json

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def load_params(jsonFile):
    with open(jsonFile) as f:
        return json.load(f)