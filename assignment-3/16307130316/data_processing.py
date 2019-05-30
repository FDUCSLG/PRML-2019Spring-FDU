import pickle
import os
import json
import re
import torch
import torch.autograd as autograd


class Dataset_batched_padding:
    def __init__(self, sl=64):  # sequence length, all padding to the length
        self.sequence_length = sl
        self.data = self.Parse_Tang_Poetry(sl)[:10000] # 10k poems
        self.train = self.data[:int(0.8*len(self.data))]
        self.develop = self.data[int(0.8*len(self.data)):]
        self.word_to_ix, self.ix_to_word = self.wordDic(self.train)  # 以训练集构造词典
        self.word_to_tensor = self.map_word_to_tensor()  # 构造字典
        self.train = self.prepare_train(self.train)
        self.develop = self.prepare_test(self.develop)

        print("VOCAB_SIZE:", len(self.word_to_ix))
        print("data_size", len(self.data))

    def wordDic(self, data):  # 增加padding
        words = sorted(set([character for sent in data for character in sent]))
        words.extend(['<EOS>', '<OOV>', '<START>', "<PAD>"])
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        idx_to_word = {idx: word for idx, word in enumerate(words)}
        # save dict for sample method
        with open('wordDic', 'wb') as f:  # save dict
            pickle.dump(word_to_idx, f)
        return word_to_idx, idx_to_word

    def Parse_Tang_Poetry(self, max_length):
        def data_clean(body):
            body = re.sub("（.*）", "", body)
            body = re.sub("{.*}", "", body)
            body = re.sub("《.*》", "", body)
            body = re.sub("\\[.*\\]", "", body)
            body = re.sub("[\d]+", "", body)
            body = re.sub("。。", "。", body)
            body = re.sub("[\\/●\\[\\]]", "", body)
            return body

        def extract_body(file, max_length):
            with open(file, encoding='utf-8') as f:
                raw = json.loads(f.read())
            res = []
            for poem in raw:
                body = ''.join(poem.get("paragraphs"))
                body = data_clean(body)
                if body != "" and len(body)+1 <= max_length and len(body) >= 10:  # 取适合大小的句子
                    res.append(body)
            return res

        src = './json/'  # change directory here
        data = []
        for filename in os.listdir(src):
            if filename.startswith("poet.tang"):
                data.extend(extract_body(src+filename, max_length))
        return data

    def map_word_to_tensor(self):  # a dict that maps word to tensor index
        word_to_tensor = {}
        for w in self.word_to_ix:
            word_to_tensor.setdefault(w, autograd.Variable(torch.LongTensor([self.word_to_ix[w]])))
        return word_to_tensor

    def prepare_train(self, data):  # toList, add <EOS>, padding
        for i in range(len(data)):
            data[i] = list(data[i])
            data[i].append("<EOS>")
            data[i].extend(["<PAD>"]*(self.sequence_length-len(data[i])))

        return data

    def prepare_test(self, data):  # toList, add <EOS>, mask with <OOV>, padding
        for i in range(len(data)):
            data[i] = list(data[i])
            for j in range(len(data[i])):
                if data[i][j] not in self.word_to_ix:
                    data[i][j] = "<OOV>"
            data[i].append("<EOS>")
            data[i].extend(["<PAD>"] * (self.sequence_length - len(data[i])))
        return data

    def makeForOneBatch(self, batch_data):  # batch_data, 包含一个batch数据的list
        In = []
        Out = []
        batch_size = len(batch_data)
        for i in range(batch_size):
            tmpIn = []
            tmpOut = []
            for j in range(1, self.sequence_length):
                w = batch_data[i][j]
                w_b = batch_data[i][j-1]
                tmpIn.append(self.word_to_tensor[w_b])
                tmpOut.append(self.word_to_tensor[w])
            tmpIn, tmpOut = torch.cat(tmpIn), torch.cat(tmpOut)
            In.append(tmpIn)
            Out.append(tmpOut)
        In = torch.cat(In).view(batch_size, -1)
        Out = torch.cat(Out).view(batch_size, -1)
        return In, Out
