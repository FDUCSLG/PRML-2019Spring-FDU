import numpy as np
import nltk, itertools, csv

TXTCODING = 'utf-8'
unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'
nltk.download('punkt')
# 解析评论文件为数值向量
class tokenFile2vector:
    def __init__(self, file_path, dict_size):
        self.file_path = file_path
        self.dict_size = dict_size

    # 将文本拆成句子，并加上句子开始和结束标志
    def _get_sentences(self):
        sents = []
        with open(self.file_path, 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            # 去掉表头
            next(reader)
            # 解析每个评论为句子
            sents = itertools.chain(*[nltk.sent_tokenize(x[0].encode('utf-8').lower()) for x in reader])
            sents = ['%s %s %s' % (start_token, sent, end_token) for sent in sents]
            print('Get {} sentences.'.format(len(sents)))

            return sents

    # 得到每句话的单词，并得到字典及字典中每个词的下标
    def _get_dict_wordsIndex(self, sents):
        sent_words = [nltk.word_tokenize(sent) for sent in sents]
        word_freq = nltk.FreqDist(itertools.chain(*sent_words))
        print('Get {} words.'.format(len(word_freq)))

        common_words = word_freq.most_common(self.dict_size-1)
        # 生成词典
        dict_words = [word[0] for word in common_words]
        dict_words.append(unknown_token)
        # 得到每个词的下标，用于生成词向量
        index_of_words = dict((word, ix) for ix, word in enumerate(dict_words))

        return sent_words, dict_words, index_of_words

    # 得到训练数据
    def get_vector(self):
        sents = self._get_sentences()
        sent_words, dict_words, index_of_words = self._get_dict_wordsIndex(sents)

        # 将每个句子中没包含进词典dict_words中的词替换为unknown_token
        for i, words in enumerate(sent_words):
            sent_words[i] = [w if w in dict_words else unknown_token for w in words]

        X_train = np.array([[index_of_words[w] for w in sent[:-1]] for sent in sent_words])
        y_train = np.array([[index_of_words[w] for w in sent[1:]] for sent in sent_words])

        return X_train, y_train, dict_words, index_of_words
