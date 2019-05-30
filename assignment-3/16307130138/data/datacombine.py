import sys
import os
import json

def get_data(data_path):
    data_set = []
    with open(data_path,'r',encoding='utf-8') as fin:
        poems = json.load(fin)
        for p in poems:
            data_set.append("".join(p['paragraphs']))
    return data_set

def get_all_data():
    base_dir="PRML\\assignment3\\data\\"
    data_set = []
    for i in range(0,58000,1000):
        print(i)
        path = base_dir+"poet.tang."+str(i)+".json"
        data_set += get_data(path)
    return data_set

if __name__ == "__main__":
    data_set = get_all_data()
    with open('PRML\\assignment3\\data\\all_poet.txt','w',encoding='utf-8') as fout:
        for data in data_set:
            fout.write(data+'\n')
    
    