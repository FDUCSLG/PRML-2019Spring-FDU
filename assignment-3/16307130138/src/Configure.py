class Config(object):
    model_prefix = "PRML\\assignment3\\checkpoints\\tang"
    load_best_model = False
    best_model_path  = "PRML\\assignment3\\checkpoints\\"
    start_words = '日红山夜湖海月'
    prefix_words = None
    gen_type= 'acrostic'
    data_path = 'PRML\\assignment3\\data\\tangshi.txt'
    #data_path = '../data/tangshi.txt'
    all_tang = False
    fastnlp = False

    n_epochs = 100
    learning_rate = 0.001
    batch_size = 32
    num_workers = 4 #加载数据时的线程数量
    num_layers = 2 #lstm层数
    embedding_dim = 128
    hidden_dim = 256
    tao = 0.6
    patience = 10
    use_gpu = False
    device = None

    save_every = 2 #每save_every个epoch存一次模型checkpoints,权重和诗,

    out_path = 'PRML\\assignment3\\out\\'
    out_potery_path = 'PRML\\assignment3\\out\\potery.txt'
    len_limit = True
    max_seq_len = 60
    max_gen_len = 60 #诗的最大长度