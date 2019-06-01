import argparse
import numpy as np
import torch
import re 
import string
import time

from sklearn.datasets import fetch_20newsgroups 
import numpy as np

def get_text_classification_datasets(small = False):
    categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc',]
    if small:
        dataset_train = fetch_20newsgroups(subset='train', categories=categories)
        dataset_test = fetch_20newsgroups(subset='test', categories=categories)
    else:
        dataset_train = fetch_20newsgroups(subset='train')
        dataset_test = fetch_20newsgroups(subset='test')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:") 
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test

def find_class_by_name(name, modules):
    """searches the provided modules for the named class and returns it"""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def load_model(model, model_path):
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    return True

def clean_data(input_data):
    # upper to lower
    input_data = (input_data).lower()
    
    # punctuation and white space
    input_data = re.sub("\d","", input_data)
    input_data = re.sub(r"[{}]+".format(string.punctuation)," ",input_data)
    input_data = re.sub(r"[{}]+".format(string.whitespace)," ",input_data)
    input_data = input_data.replace('\\', " ")
    # stop words
    # number
    # split
    output_data = re.split(" ", input_data)
    output_data.remove('')
    return output_data

def clean_dataset(input_dataset):
    clean_dataset = []
    input_dataset = np.array(input_dataset)
    for i, example in enumerate(input_dataset):
        temp = clean_data(example)
        clean_dataset.append(temp)
    return clean_dataset  

def draw_line(num):
    for i in range(num):
        print("-",end="")
    print("\n", end="")

def box(sentence, other=""):
    sentence = "| " + sentence + " |"
    length = len(sentence)
    draw_line(length)
    print(sentence, end="")
    print(other)
    draw_line(length)

def box_dict(sentence_dict, title=None):
    sentence_list = []
    for i in sentence_dict:
        temp = str(i)
        temp += " : "
        temp += str(sentence_dict[i])
        sentence_list.append(temp)
    length = max([len(i) for i in sentence_list])
    if title is not None:
        print(title)
    draw_line(length+4)
    for i in sentence_list:
        dif_length = length - len(i)
        print("| " + i + " "*dif_length + " |")
    draw_line(length+4)


