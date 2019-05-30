from torch.utils.data.dataset import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class HANDataset(Dataset):
    def __init__(self, dataset, dict, sent_length=30, word_length=35):
        super(HANDataset, self).__init__()

        self.texts = dataset.texts
        self.labels = dataset.labels
        self.dict = dict[:,0]
        self.dict = self.dict.reshape(len(self.dict))
        self.dict = self.dict.tolist()
        self.sent_length = sent_length
        self.word_length = word_length
        print(len(self.labels),len(self.texts))
        self.num_calsses = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [[self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(
            text=sentence)] for sentence in sent_tokenize(text=text)]

        for sentence in document_encode:
            if len(sentence) < self.word_length:
                extended_words = [-1 for _ in range(self.word_length - len(sentence))]
                sentence.extend(extended_words)

        if len(document_encode) < self.sent_length:
            extended_sent = [[-1 for _ in range(self.word_length)] for _ in range(self.sent_length - len(document_encode))]
            document_encode.extend(extended_sent)

        document_encode = [sentence[: self.word_length] for sentence in document_encode][: self.sent_length]
        
        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label
