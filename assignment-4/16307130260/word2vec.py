import pickle
import numpy as np

from data import read_data
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from fastNLP import Vocabulary, Const


def train_and_save():
    (vocab, train_data, dev_data, test_data) = read_data()
    train_data.apply_field(lambda x: [ vocab.to_word(i) for i in x ],
                            field_name=Const.INPUT, new_field_name="words_for_train")

    # print(train_data["words_for_train"])
    sentences = train_data["words_for_train"]

    path = get_tmpfile("word2vec.model")
    model = Word2Vec(sentences, size=128, window=64, min_count=1, workers=4)
    model.build_vocab(sentences, update=True)
    model.train(sentences=sentences, total_examples=len(sentences),
        epochs=model.iter, compute_loss=True)

    for e in model.most_similar(positive=["on"], topn=10):
        print(e[0], e[1])

    model.save("word2vec.model")

def get_w():
    (vocab, _, _, _) = read_data()

    model = Word2Vec.load("word2vec.model")
    w = []
    w.append(np.array([0] * 128).astype(np.float32))
    for i in range(len(vocab)-1):
        w.append(np.array(model.wv[vocab.to_word(i+1)]).astype(np.float32))
    print(len(w))
    print(w[0])
    pickle.dump(w, open("weight.bin", "wb"))

if __name__ == "__main__":
    # train_and_save()
    get_w()