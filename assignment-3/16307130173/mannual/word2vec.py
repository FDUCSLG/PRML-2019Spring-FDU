import numpy as np

class word2Vec:
    def __init__(self, file_path):
        
        self.word_dict = {'~': 0, '*': 1}
        self.freq_dict = {'~': 0}
        self.dict_size = 2
        self.max_sentense_size = 0
        self.X = []
        self.y = []
        self.get_dict(file_path)

    def get_dict(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        for index, line in enumerate(f):
            if index % 2 == 0:
                continue
            stt = line.strip('\n')
            if not stt:
                break
            length = len(stt)
            if length <= 100:
                self.max_sentense_size = max(self.max_sentense_size, length)

                for i in range(length - 1):
                    if stt[i] not in self.freq_dict:
                        self.freq_dict[stt[i]] = 1
                    else:
                        self.freq_dict[stt[i]] += 1

                self.freq_dict['~'] += 1

        f = open(file_path, 'r', encoding='utf-8')
        for index, line in enumerate(f):
            if index % 2 == 0:
                continue
            stt = line.strip('\n')
            if not stt:
                break
            length = len(stt)
            if length <= 100:
                lst_x = []
                lst_y = []
                for i in range(length - 1):
                    if stt[i] not in self.word_dict:
                        if self.freq_dict[stt[i]] > 15:
                            self.word_dict[stt[i]] = self.dict_size
                            self.dict_size += 1
                        else:
                            self.word_dict[stt[i]] = 1

                for i in range(length - 1):
                    lst_x.append(self.word_dict[stt[i]])
                    lst_y.append(self.word_dict[stt[i + 1]])

                lst_y[length - 2] = 0

                self.X.append(lst_x + [0 for i in range(length, self.max_sentense_size)])
                self.y.append(lst_y + [0 for i in range(length, self.max_sentense_size)])
                
        self.index_dict = {}
        for k, v in self.word_dict.items():
            self.index_dict[v] = k

    def get_dataset(self):
        return np.array(self.X), np.array(self.y)
