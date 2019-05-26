import os
import re
import numpy as np
from fastNLP import Vocabulary

def pre_process(file_name):
    
    poem = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index % 2 == 1:
                raw_line = line.strip()

                raw_line = re.sub('，', '', raw_line)
                raw_line = re.sub('。', '', raw_line)

                length = len(raw_line)
                if length < 100:
                    raw_line = raw_line + '~' * (100 - length)
                
                poem.append(raw_line[:100])                
        
    word_dict = Vocabulary()
    for line in poem:
        for character in line:
            word_dict.add(character)
            
    word_dict.build_vocab()

    data = []
    for pi in poem:
        p = []
        for ch in pi:
            p.append(word_dict.to_index(ch))
        data.append(p)
    data = np.array(data)
    
    return word_dict, data

