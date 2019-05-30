import numpy as np
import codecs

def read_pretrained_embeddings(filename, w2i,embedding_dim=100):
    word_to_embed = {}
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            split = line.split()
            if len(split) > 2:
                word = split[0]
                if word in w2i:
                    vec = split[1:]
                    word_to_embed[word] = vec

    pre=np.std(np.array(list(word_to_embed.values())).astype("float"))
    print("in:",len(word_to_embed),len(w2i),pre)
    out = np.random.uniform(-0.8, 0.8, (len(w2i), embedding_dim))
    for word, embed in word_to_embed.items():
        out[w2i[word]] = np.array(embed)
    return out
    
def to_id_list(w2i):
    i2w = [None] * len(w2i)
    for w, i in w2i.items():
        i2w[i] = w
    return i2w
