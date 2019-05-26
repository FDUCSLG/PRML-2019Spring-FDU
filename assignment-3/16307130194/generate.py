from model import PoetryModel
from config import Config
from dataset import get_dataset

import os
import torch
from torch.autograd import Variable


def generate_poem(model, vocabulary, max_gen_len, temperature, start_word=None):
    input = Variable(torch.Tensor(([vocabulary.to_index('<START>')]))).view(1, 1).long()
    hidden, cell = None, None

    results = start_word
    start_word_len = len(start_word)
    size = 0
    for i in range(max_gen_len):
        output, hidden, cell = model(input, hidden, cell).values()

        if i < start_word_len:
            w = results[i]
            input = input.data.new([vocabulary.to_index(w)]).view(1, 1)
        else:
            output = output.data.view(-1).div(temperature).exp()
            for _ in range(10):
                index = torch.multinomial(output, 1)[0]
                w = vocabulary.to_word(int(index))
                if w != "<oov>" and w != "<START>" and w != "<pad>":
                    break
            results += w
            input = input.data.new([index]).view(1, 1)
        if w == '。' or w == '?':
            size += 1
        if w == '<eos>' or size >= max_gen_len:
            break

    return [results]


def generate():
    config = Config()
    train_data, dev_data, vocabulary = get_dataset(config.data_path)
    poetry_model = PoetryModel(vocabulary_size=len(vocabulary), embedding_size=config.embedding_size,
                               hidden_size=config.hidden_size)
    states = torch.load(os.path.join(config.save_path, config.model_name)).state_dict()
    poetry_model.load_state_dict(states)

    for start_word in config.start_words:
        result = generate_poem(model=poetry_model, start_word=start_word,
                               vocabulary=vocabulary,
                               max_gen_len=config.max_gen_len,
                               temperature=0.6)
        print(result)


if __name__ == "__main__":
    generate()
