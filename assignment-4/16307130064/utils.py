import fastNLP
from fastNLP import Instance,Vocabulary, DataSet, Const

import string

def deal(r):
    for sp in string.punctuation:
        r=r.replace(sp,"")
    for sp in string.whitespace:
        r=r.replace(sp," ")
    return r

def make_dataset(data):
    dataset=DataSet()
    mx=0
    le=None
    for x,y in zip(data.data,data.target):
        xx=deal(x)
        ins=Instance(sentence=xx, label=int(y))
        if mx<len(xx.split()):
            mx=max(mx,len(xx.split()))
            le=xx
        dataset.append(ins)
    print(mx)
    dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')   
    dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    
    dataset.rename_field('words', Const.INPUT)
    dataset.rename_field('seq_len', Const.INPUT_LEN)
    dataset.rename_field('label', Const.TARGET)
    
    dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    dataset.set_target(Const.TARGET)
    return dataset
