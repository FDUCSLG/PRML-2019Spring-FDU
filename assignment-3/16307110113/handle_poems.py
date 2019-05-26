# -*- coding: utf-8 -*-
import os
os.sys.path.append('..')
import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from torch import nn

import collections
import numpy as np
import string
import random

hanzi_punctuation = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
start_token = 'S'
end_token = 'E'
# 超参数
vocab_size = 6000
sentence_len = 48

print("vocab_size:", vocab_size)
print("sentence_len:", sentence_len)

def process_poems(file_name, sentence_len, vocab_size):
    sentences = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    # content = line.replace(' ', '').replace('，','').replace('。','')
                    content = line.replace(' ', '') #包含标点符号
                    if len(content) < 10 or len(content) > sentence_len:
                        continue
                    # print(content)
                    content = content + end_token
                    sentences.append(content)
            except ValueError as e:
                pass

    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        instance['target'] = sentence[1:] + sentence[-1]
        dataset.append(instance)

    dataset.set_input("raw_sentence")
    dataset.set_target("target")

    # for iter in dataset:
    #     print(iter)
    print("dataset_size:", len(dataset))

    train_data, dev_data = dataset.split(0.2)
    train_data.rename_field("raw_sentence", "sentence")
    dev_data.rename_field("raw_sentence", "sentence")
    vocab = Vocabulary(max_size=vocab_size, min_freq=2, unknown='<unk>', padding='<pad>')

    # 构建词表
    train_data.apply(lambda x: [vocab.add(word) for word in x['sentence']])
    vocab.build_vocab()
    print("vocabulary_size:", len(vocab))

    # 根据词表index句子
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['sentence']], new_field_name='sentence')
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['sentence']], new_field_name='sentence')
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')

    return train_data, dev_data, vocab


def process_poems_large(file_name, sentence_len, vocab_size):
    sentences = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    title, content = line.split(':')
                    # print(title)
                    # print(content)
                    # content = line.replace(' ', '').replace('，','').replace('。','')
                    content = content.replace(' ', '') #包含标点符号
                    # 可以只取五言诗
                    # if len(content) < 6 or content[5] != '，':
                    #     continue
                    if len(content) < 20:
                        continue
                    if ':' in content or '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                        continue
                    #截断长度
                    if len(content) > sentence_len:
                        content = content[:sentence_len]
                    content = content + end_token
                    sentences.append(content)
            except ValueError as e:
                pass

    dataset = DataSet()
    # sentences = random.sample(sentences, 5000)
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        instance['target'] = sentence[1:] + sentence[-1]
        dataset.append(instance)

    dataset.set_input("raw_sentence")
    dataset.set_target("target")
    
    # for iter in dataset:
    #     print(iter['raw_sentence'])
    print("dataset_size:", len(dataset))

    train_data, dev_data = dataset.split(0.2)
    train_data.rename_field("raw_sentence", "sentence")
    dev_data.rename_field("raw_sentence", "sentence")
    vocab = Vocabulary(max_size=vocab_size, min_freq=2, unknown='<unk>', padding='<pad>')

    # 构建词表
    train_data.apply(lambda x: [vocab.add(word) for word in x['sentence']])
    vocab.build_vocab()

    # 根据词表index句子
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['sentence']], new_field_name='sentence')
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['sentence']], new_field_name='sentence')
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')

    print("vocabulary_size:", len(vocab))

    return train_data, dev_data, vocab


def save_vocab(file_name, vocab):
    with open(file_name,'w') as f:
        for i in range(len(vocab)):
            word = vocab.to_word(i)
            if word != '<unk>' and word != '<pad>':
                f.writelines(word)   

# train_data, dev_data, vocab = process_poems('../handout/tangshi.txt', sentence_len, vocab_size)
train_data, dev_data, vocab = process_poems_large('poems.txt', sentence_len, vocab_size)
save_vocab("vocab.txt", vocab)
train_data.save("train_data.txt")
dev_data.save("dev_data.txt")

