
class Config(object):
    data_path = './tangshi.txt'
    # data_path = '../handout/tangshi.txt'
    save_path = './model_log/'
    model_name = 'best_PoetryModel_PPL_2019-05-26-12-02-05'
    loss_path = './loss_log/'
    loss_name = '0001.txt'
    max_gen_len = 128
    temperature = 0.6
    device = 'cuda:4'
    check_code_level = 0
    sequence_length = 128
    embedding_size = 128
    hidden_size = 512
    epoch = 256
    batch_size = 128
    print_every = 10
    validate_every = 50
    optimizer = 'adam'
    lr = 1e-3
    weight_decay = 0
    momentum = 0.9
    rho = 0.9
    eps = 1e-6
    patience = 10
    start_words = ['日', '红', '山', '夜', '湖', '海', '月']
