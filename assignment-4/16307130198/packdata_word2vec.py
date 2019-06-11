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
import gensim
from gensim.models import word2vec


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

def transform_word2vec(dataset):
    sentence_list = []
    for i in dataset:
        sentence_list.append(i['words'])
    model = word2vec.Word2Vec(sentence_list, hs=1,min_count=1,window=64,size=256) 
    return model

def gain_ew(idx2word, word2vec_model):
    v_len = len(idx2word)
    import numpy as np
    embedding_weight = np.zeros((v_len, 256))

    for index in range(v_len):
        if index != 0:
            temp_word = idx2word[index]
            embedding_weight[index] = (word2vec_model.wv[temp_word])
            
    return embedding_weight 

if __name__ == '__main__':
    dataset_train, dataset_test = utils.get_text_classification_datasets(small=False)
    categories = dataset_train.target_names
    
    dataset = ndarray2dataSet(dataset_train.data, dataset_train.target)
    test_data = ndarray2dataSet(dataset_test.data, dataset_test.target)    

    # construct vocabulary table
    vocab = Vocabulary(min_freq=10).from_dataset(dataset, field_name='words')
    vocab.index_dataset(dataset, field_name='words',new_field_name='input_data') 
    vocab.index_dataset(test_data, field_name='words',new_field_name='input_data')
    
    idx2word = vocab.idx2word  
    
    dataset.apply(lambda x: [idx2word[idx] for idx in x['input_data']], new_field_name="words")
    
    word2vec_model = transform_word2vec(dataset)
    embedding_weight = gain_ew(idx2word, word2vec_model)

    #exit()
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
    pickle.dump(embedding_weight, open("./dataset/embedding_weight.pkl", "wb"), 2)



