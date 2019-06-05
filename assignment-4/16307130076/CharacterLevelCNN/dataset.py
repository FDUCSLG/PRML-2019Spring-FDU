import numpy as np
import string
from torch.utils.data import Dataset

class CharacterDataset(Dataset):
    def __init__(self, dataset, max_length=1014):
        super(CharacterDataset, self).__init__()
        self.vocabulary = list(
            """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        self.texts = dataset.texts
        self.labels = dataset.labels
        self.max_length = max_length
        self.length = len(self.texts)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index].lower()
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(
                self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
