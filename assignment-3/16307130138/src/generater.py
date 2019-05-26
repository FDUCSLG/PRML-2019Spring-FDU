import numpy as np
import sys
sys.path.append('../')
from model import *

def generate_poet(model, start_words, vocab, conf):
    if not isinstance(start_words, list):
        results = list(start_words)
    else:
        results = start_words
    start_len = len(start_words)
    input = Variable(torch.Tensor([vocab.to_index('<START>')]).view(1,1).long())
    if conf.use_gpu:input=input.cuda()
    hidden = None
    if conf.prefix_words:
        for word in conf.prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([vocab.to_index(word)])).view(1,1)
    
    for i in range(conf.max_gen_len):
        output,hidden = model(input,hidden)
        if i < start_len:
            next_word = results[i]
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        else:
            next_word_id = output.argmax(dim = 1)
            # prob = torch.exp(output[0]/conf.tao)
            # next_word_id = prob.multinomial(1)
            
            # next_word = vocab.to_word(next_word_id.cpu().numpy().tolist()[0])
            next_word = vocab.to_word(next_word_id.item())
            results.append(next_word)
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        if next_word == '<EOS>':
            del results[-1]
            break
    print("".join(results))
    return results

def gen_acrostic(mode,start_words,vocab,conf):
    results = []
    start_word_len = len(start_words)
    input = Variable(torch.Tensor([vocab.to_index('<START>')]).view(1,1).long())
    if conf.use_gpu:
        input=input.cuda()
    hidden = None
    index=0 # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word='<START>'
 
    if conf.prefix_words:
        for word in conf.prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([vocab.to_index(word)])).view(1,1)
 
    for i in range(conf.max_gen_len):
        output,hidden = model(input,hidden)
        top_index  = output.data[0].topk(1)[1][0]
        w = vocab.to_word(top_index.cpu().numpy().tolist()[0])
 
        if (pre_word  in {u'。',u'！','<START>'} ):
            # 如果遇到句号，藏头的词送进去生成
 
            if index==start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:  
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index+=1
                input = Variable(input.data.new([vocab.to_index(w)])).view(1,1)    
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = Variable(input.data.new([vocab.to_index(w)])).view(1,1)
        results.append(w)
        pre_word = w
    return results


def gen(**kwargs) :
    conf = Config()
    for k,v in kwargs.items():
        setattr(conf,k,v)
    pdata = PoemData()
    pdata.read_data(conf)
    pdata.get_vocab()
    model = MyPoetryModel(pdata.vocab_size, conf.embedding_dim,conf.hidden_dim)
    map_location = lambda s,l:s
	# 上边句子里的map_location是在load里用的，用以加载到指定的CPU或GPU，
	# 上边句子的意思是将模型加载到默认的GPU上
    state_dict = torch.load(conf.model_path, map_location = map_location)
    model.load_state_dict(state_dict)

    if conf.use_gpu:
        model.cuda()
    if sys.version_info.major == 3:
        if conf.start_words.insprintable():
            print(conf.start_words)
            print(conf.prefix_words)
        else:
            conf.start_words = conf.start_words.encode('ascii','surrogateescape').decode('utf8')
            conf.prefix_words = conf.prefix_words.encode('ascii','surrogateescape').decode('utf8') if conf.prefix_words else None
        start_words = start_words.replace(',',u'，').replace('.',u'。').replace('?',u'？')
        gen_poetry = gen_acrostic if conf.gen_type=='acrostic' else generate_poet
        result = gen_poetry(model,start_words,pdata.vocab,conf)
        print(''.join(result))
