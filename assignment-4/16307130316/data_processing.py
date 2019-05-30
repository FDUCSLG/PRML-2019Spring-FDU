from sklearn.datasets import fetch_20newsgroups
import fastNLP
import string
import torch
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from gensim.models import Word2Vec


def get_text_classification_datasets():
    categories = ['comp.os.ms-windows.misc', 'misc.forsale', 'rec.motorcycles', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.misc', 'talk.religion.misc']
    dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
    dataset_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test


def split_sent(line):
    line = line.lower()
    for c in string.punctuation:
        line = line.replace(c, "")
    for w in string.whitespace:
        line = line.replace(w, " ")
    line = line.split()
    return line


def get_fastnlp_dataset():
    text_train, text_test = get_text_classification_datasets()
    train_data = DataSet()
    test_data = DataSet()
    for i in range(len(text_train.data)):
        train_data.append(Instance(text=split_sent(text_train.data[i]), target=int(text_train.target[i])))
    for i in range(len(text_test.data)):
        test_data.append(Instance(text=split_sent(text_test.data[i]), target=int(text_test.target[i])))

    # 构建词表
    vocab = Vocabulary(min_freq=5, unknown='<unk>', padding='<pad>')
    train_data.apply(lambda x: [vocab.add(word) for word in x['text']])
    vocab.build_vocab()

    # 根据词表映射句子
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name='word_seq')
    test_data.apply(lambda x: [vocab.to_index(word) for word in x['text']], new_field_name='word_seq')

    # 设定特征域和标签域
    train_data.set_input("word_seq")
    test_data.set_input("word_seq")
    train_data.set_target("target")
    test_data.set_target("target")

    return train_data, test_data, vocab


def get_word2vec(embed_size=50):
    categories = ['comp.os.ms-windows.misc', 'misc.forsale', 'rec.motorcycles', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.misc', 'talk.religion.misc']
    dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
    sentences = []
    for sent in dataset_train.data:
        sentences.append(split_sent(sent))
    model = Word2Vec(sentences, sg=1, size=embed_size, window=3)
    model.save('word2vec')

# get_word2vec()
# model = Word2Vec.load('word2vec')
# print(model['a'])
# print(model.similarity('woman','man'))


def get_pretrained_weight(vocab, embed_size=50):
    get_word2vec(embed_size=embed_size)
    model = Word2Vec.load('word2vec')
    weight = torch.zeros(len(vocab.idx2word), embed_size)
    for i in range(len(model.wv.index2word)):
        try:
            index = vocab.word2idx[model.wv.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(model.wv.get_vector(vocab.idx2word[vocab.word2idx[model.wv.index2word[i]]]))

    return weight

def load_model(model, model_path):
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    return model

