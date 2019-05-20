import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import argparse
import numpy as np

import re 
import string
import time

def find_class_by_name(name, modules):
    """searches the provided modules for the named class and returns it"""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

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
    for i, example in enumerate(input_dataset):
        temp = clean_data(example)
        clean_dataset.append(temp)
    return clean_dataset  

def build_mapping_dict(clean_dataset):
    record_dict = {}
    mapping_dict = {}
    for example in clean_dataset:
        for word in example:
            if word in record_dict:
                record_dict[word]+=1
            else:
                record_dict[word]=1

    index = 1
    for i in record_dict:
        if record_dict[i]>=10:
            mapping_dict[i] = index
            index+=1
    
    return mapping_dict

def data2vec(clean_dataset, mapping_dict):
    feature_vector = []
    
    # mappting_dict contains all the mapping information, we can ultilize it to trainsform the clean_data into vectors
    vec_length = len(mapping_dict)
    
    for temp_data in clean_dataset:
        init_feature_vec = np.append(np.array([1]), np.zeros(vec_length))
        for word in temp_data:
            if word in mapping_dict:
                init_feature_vec[mapping_dict[word]] = 1

        feature_vector.append(init_feature_vec)
    
    return np.array(feature_vector)


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


