import sys
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import _pickle as pickle 
import os
import utils
import re
import string
import _pickle as pickle

def ndarray2dataSet(data, target):
    dataset = DataSet()
    for i,input_data in enumerate(data):
        input_data = (input_data).lower()

        # punctuation and white space
        
        input_data = re.sub("\d","", input_data)
        
        input_data = re.sub(r"[{}]+".format(string.punctuation)," ",input_data)
        input_data = re.sub(r"[{}]+".format(string.whitespace)," ",input_data)
        
        input_data = input_data.replace('\\', " ")
        input_data = re.split(" ", input_data)
        input_data.remove('')
        dataset.append(Instance(target = int(target[i]), words = input_data))

    return dataset

if __name__ == '__main__':
    dataset_train, dataset_test = utils.get_text_classification_datasets(small=False)
    categories = dataset_train.target_names
    
    dataset = ndarray2dataSet(dataset_train.data, dataset_train.target)
    test_data = ndarray2dataSet(dataset_test.data, dataset_test.target)    

    # construct vocabulary table
    vocab = Vocabulary(min_freq=10).from_dataset(dataset, field_name='words')
    vocab.index_dataset(dataset, field_name='words',new_field_name='input_data') 
    vocab.index_dataset(test_data, field_name='words',new_field_name='input_data')
    
    dataset.set_input("input_data")
    dataset.set_target("target")
    test_data.set_input("input_data")
    test_data.set_target("target")
    
    train_data, validate_data = dataset.split(0.2)

    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    pickle.dump(train_data, open("./dataset/train_data.pkl", "wb"), 2)
    pickle.dump(validate_data, open("./dataset/validate_data.pkl", "wb"), 2)
    pickle.dump(test_data, open("./dataset/test_data.pkl", "wb"), 2)
    pickle.dump(vocab, open("./dataset/vocab.pkl", "wb"), 2)    





