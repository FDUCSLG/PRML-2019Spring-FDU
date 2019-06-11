import os


class Config:
    # path
    data_path = "./"
    train_name = "train_data.pkl"
    dev_name = "dev_data.pkl"
    test_name = "test_data.pkl"
    vocabulary_name = "vocabulary.pkl"
    weight_name = "weight.pkl"

    # cnn
    embed_dim = 128
    kernel_sizes = (3, 4, 5)
    kernel_num = 100
    in_channels = 1
    dropout = 0.5
    static = False

    # rnn
    num_layers = 1
    hidden_dim = 256

    # Adam
    lr = 1e-3
    weight_decay = 0

    # early stop
    patience = 20

    # train
    device = [2]
    epoch = 128
    batch_size = 64
    print_every = 10
    validate_every = 100

    # task
    class_num = 20
    task_name = "cnn_w2v"
    save_path = "./model_log"

