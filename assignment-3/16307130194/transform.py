import numpy as np
from langconv import *


data = np.load('tang.npz')
data, ix2word = data['data'], data['ix2word'].item()
poetry = []
for sentence in data:
    text = ''
    for d in text:
        text += ix2word[d]
    poetry.append(text)


with open('tangshi.txt', 'w') as f:
    for i, line in poetry:
        line = Converter('zh-hans').convert(line.decode('utf-8')).encode('utf-8')
        if i < length - 1:
            f.write(line + '\n\n')
        else:
            f.write(line)
