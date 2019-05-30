import torch


def create_input_tensor(poetry, word_list):
    tensor = torch.zeros(len(poetry) - 1).long()
    for i in range(0, len(poetry) - 1):
        word = poetry[i]
        tensor[i] = word_list.index(word)
    return tensor

def create_target_tensor(poetry, word_list):
    tensor = torch.zeros(len(poetry) - 1).long()
    for i in range(1, len(poetry)):
        word = poetry[i]
        tensor[i - 1] = word_list.index(word)
    return tensor

def create_training_data(poetry, word_list):
    input_tensor = create_input_tensor(poetry, word_list)
    target_tensor = create_target_tensor(poetry, word_list)
    return input_tensor, target_tensor