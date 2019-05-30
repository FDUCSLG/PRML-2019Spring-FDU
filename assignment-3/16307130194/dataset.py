from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from config import Config

config = Config()


def get_dataset(data_path):
    print('Getting dataset...')

    poetry = []
    with open(data_path, 'r', encoding='utf-8') as f:
        poem = ''
        for line in f:
            if len(line) <= 1:
                ins = Instance(text=poem)
                if len(poem) > 10:
                    poetry.append(ins)
                poem = ''
            else:
                poem += line.strip('\n')
    # print(poetry[0])

    data = DataSet(data=poetry)
    print("Original data:", data[0])

    vocabulary = Vocabulary(min_freq=2, unknown='<oov>', padding='<pad>')
    vocabulary.add_word('<eos>')
    vocabulary.add_word('<START>')
    data.apply(lambda x: [vocabulary.add(char) for char in x['text']])
    vocabulary.build_vocab()
    print('pad:', vocabulary.to_index('<pad>'))
    print('Vocab size:', len(vocabulary))

    data.apply(lambda x: [vocabulary.to_index(char) for char in x['text']], new_field_name='text')
    data.apply(lambda x: [vocabulary.to_index('<START>')] + x['text'] + [vocabulary.to_index('<eos>')], new_field_name='text')
    data.apply(lambda x: x['text'][0:min(config.sequence_length, len(x['text']))], new_field_name='text')
    data.apply(lambda x: [vocabulary.to_index('<pad>')] * (config.sequence_length - len(x['text'])) + x['text'], new_field_name='text')
    data.apply(lambda x: x['text'][0:-1], new_field_name='input')
    data.apply(lambda x: x['text'][1:], new_field_name='target')
    data.set_input('input')
    data.set_target('target')

    # length = config.sequence_length
    # for i, d in enumerate(data):
    #     if length != len(d['text']):
    #         print("wrong!")
    # exit()

    train_data, dev_data = data.split(0.2)
    print('Train data size:', len(train_data))
    print('Dev data size:', len(dev_data))
    print("Train data:", train_data[20])
    # print("Dev data:", dev_data[0])

    return train_data, dev_data, vocabulary
