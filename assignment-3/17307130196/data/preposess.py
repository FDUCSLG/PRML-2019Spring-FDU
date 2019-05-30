import json
import sys
sys.path.append("..")
from data.langconv import *
import random 
def build_dict(addr="data/poems/"):
    char_ct=[] # 记录字符以及出现的次数
    dictionary_id={} # 记录每一个字对应的id
    ct=0
    for i in range(50):
        addr_=addr+'poet.tang.'+str(i*1000)+'.json'
        with open(addr_,"r",encoding='UTF-8') as f:
            all = json.loads(f.read())
            for poem in all:
                content=poem['paragraphs']
                if len(content)==4:
                    flag=1
                    if len(content[0])==12:
                        for line in content: 
                            if len(line)!=12:
                                flag=0
                                break   
                            line = Converter('zh-hans').convert(line)
                            for j in range(0,12):
                                in_char=line[j]
                                if in_char not in dictionary_id.keys():
                                    char_ct.append([in_char,1])
                                    dictionary_id[in_char]=ct
                                    ct+=1
                                else:
                                    char_ct[dictionary_id[in_char]][1]+=1
                            if not flag:
                                break
    def by_ct(t):#传入的参数是tuple
        return t[1] #把tuple中的第一个元素取出来
    
    sorted_ct=sorted(char_ct,key=by_ct,reverse=True)
    
    dictionary={}
    dictionary_rev={}
    D_dict=2000
    for i in range(D_dict):
        dictionary[sorted_ct[i][0]]=i
        dictionary_rev[i]=sorted_ct[i][0]
    
    dictionary['*']=D_dict
    dictionary_rev[D_dict]='*'
    dictionary['$']=D_dict+1
    dictionary_rev[D_dict+1]='$'


    return dictionary, dictionary_rev

def clr_poem(dictionary,dictionary_rev,addr="data/poems/"):
    eight_lines=[]
    unknown=0
    for i in range(50):
        addr_=addr+'poet.tang.'+str(i*1000)+'.json'        
        with open(addr_,"r",encoding='UTF-8') as f:
            all = json.loads(f.read())
            for poem in all:
                content=poem['paragraphs']
                if len(content)==4:
                    chars=[]
                    chars.append('*')
                    flag=1
                    if len(content[0])==12:
                        for line in content: 
                            if len(line)!=12:
                                flag=0
                                break   
                            line = Converter('zh-hans').convert(line)
                            # print(line)
                            for j in range(0,12):
                                in_char=line[j]
                                if in_char not in dictionary.keys():
                                    id=random.randint(0,len(dictionary)-1)
                                    in_char=dictionary_rev[id]
                                    unknown+=1
                                chars.append(in_char)
                            if not flag:
                                break
                        if flag:
                            chars.append('$')
                            eight_lines.append(chars)
    print('Unknown chars count: %f'%(float(unknown)/float(len(eight_lines)*4*12)))
    num=len(eight_lines)
    train_set=eight_lines[0:int(num*3/4)]
    devel_set=eight_lines[int(num*3/4):-1]
    return train_set,devel_set

def build_pretrain_dict():
    dict={}
    with open("data/embedding.word","r",encoding='UTF-8') as f:
        f.readline()
        while 1:
            line=f.readline()
            if not line:
                break
            split=line.split()
            char=split[0]
            
            vec=[0]*300 #fot the dimension of input
            for i in range(300):
                vec[i]=float(split[i+1])
            dict[char]=vec
        f.close
    return dict


def reform(poem):
    ct=1
    poem_=[]
    poem_.append(poem[0])
    print(poem[0])
    for i in range(4):
        str=""
        for j in range(12):
            str+=poem[ct]
            ct+=1
        poem_.append(str)
        print(str)
    poem_.append(poem[-1])
    print(poem[-1])
    return poem_    


def write(addr,list):
    with open(addr,"w",encoding='UTF-8') as f:
        for num in list:
            f.write(str(num))
            f.write(' ')


def load(addr):
    with open(addr,"r",encoding='UTF-8') as f:
        line=f.readline()
        list_char=line.split()
        list_num=[]
        for num in list_char:
            list_num.append(float(num))
    return list_num