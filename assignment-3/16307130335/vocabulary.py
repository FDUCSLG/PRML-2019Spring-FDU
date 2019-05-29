# coding=UTF-8
from fastNLP import Vocabulary, DataSet, Instance
import json


def get_dataset():
    dataset = DataSet()
    for i in range(0, 58000, 1000):
        file = "json/poet.tang." + str(i)+".json"
        with open(file, 'r') as load_f:
            load_dict = json.load(load_f)
            for poem in load_dict:
                data = ""
                if poem['paragraphs']!=[]:
                    data = poem['paragraphs'][0]
                data2 = ""
                for s in data:
                    if s != "," and s != "。" and s != "，":
                        data2 += s
                dataset.append((Instance(poem=data2)))
    train_data, dev_data = dataset.split(0.2)
    return train_data, dev_data


def get_vocabulary(train_data, test_data):
    # 构建词表, Vocabulary.add(word)
    vocab = Vocabulary(min_freq=0, unknown='<unk>', padding='<pad>')
    train_data.apply(lambda x: [vocab.add(word) for word in x['poem']])
    vocab.build_vocab()
    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['poem']], new_field_name='words')
    test_data.apply(lambda x: [vocab.to_index(word) for word in x['poem']], new_field_name='words')
    
    return vocab, train_data, test_data



def main():
    with open('model_0.json', 'r') as f:
        j = json.load(f)

if __name__ == "__main__":
    main()