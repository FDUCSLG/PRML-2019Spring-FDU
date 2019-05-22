import numpy as np
from langconv import *


data = np.load('tang.npz')
data, ix2word = data['data'], data['ix2word'].item()
print(data[0])
print(ix2word[data[0][-3]])

poetry = []
for sentence in data:
    text = ''
    for d in text:
        text += ix2word[d]
    poetry.append(text)

print(poetry[0])

with open('tangshi.txt', 'w', encoding='utf-8') as f:
    length = len(poetry)
    for i, line in enumerate(poetry):
        line = Converter('zh-hans').convert(line)
        if i < length - 1:
            f.write(line + '\n\n')
        else:
            f.write(line)
