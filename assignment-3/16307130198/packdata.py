import sys
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import _pickle as pickle 
import os

fix_length = 120

def padding(ins):
    item = ins['poetry']
    # to list
    item = list(item)
    item = ['<start>'] + item + ['<end>']
    item_length = len(item)

    # fix length
    if item_length > fix_length:
        item = item[0:fix_length]
    elif item_length < fix_length:
        pad_length = fix_length - item_length
        pad_str_list = pad_length * ["<pad>"]
        item = pad_str_list + item
    return list(item)

def gainTarget(ins):
    item = ins['poetry']
    return item[1:]

def gainInput(ins):
    item = ins['poetry']
    return item[0:-1]

if __name__ == '__main__':
    dataset = DataSet.read_csv('./poetry.csv', headers=['poetry'], sep='\t')
    """
    print(len(dataset))
    max_len = 0
    for i in dataset:
        temp_len = len(i['poetry'])
        if temp_len > max_len:
            max_len = temp_len
    print(max_len)
    exit()
    """
    dataset.apply(padding, new_field_name="poetry")
    
    # construct target
    dataset.apply(gainTarget, new_field_name="target") 
    # construct input
    dataset.apply(gainInput, new_field_name="input_data")
    

    train_data, validate_data = dataset.split(0.2)
    # construct vocabulary table
    vocab = Vocabulary(min_freq=2)
    train_data.apply(lambda x: [vocab.add(word) for word in x['poetry']])
    vocab.build_vocab()
    
    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['poetry']], new_field_name='poetry')
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['input_data']], new_field_name='input_data')
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')

    validate_data.apply(lambda x: [vocab.to_index(word) for word in x['poetry']], new_field_name='poetry')
    validate_data.apply(lambda x: [vocab.to_index(word) for word in x['input_data']], new_field_name='input_data')
    validate_data.apply(lambda x: [vocab.to_index(word) for word in x['target']], new_field_name='target')
    
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    pickle.dump(train_data, open("./dataset/train_data.pkl", "wb"), 2)
    pickle.dump(validate_data, open("./dataset/validate_data.pkl", "wb"), 2)
    pickle.dump(vocab, open("./dataset/vocab.pkl", "wb"), 2)    





