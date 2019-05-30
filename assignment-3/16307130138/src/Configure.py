class Config(object):
    model_prefix = "../checkpoints/tang"
    load_best_model = False
    best_model_path  = "../checkpoint/best_train_model"
    start_words = '日红山夜湖海月'
    prefix_words = None
    gen_type= 'acrostic'
    data_path = '../handout/tangshi.txt'
    all_tang = False
    n_epochs = 200
    learning_rate = 1e-3
    batch_size = 8
    num_workers = 8 #加载数据时的线程数量
    num_layers = 2 #lstm层数
    embedding_dim = 256
    hidden_dim = 256
    tao = 0.8
    patience = 10
    use_gpu = True
    device = None

    save_every = 2 #每save_every个epoch存一次模型checkpoints,权重和诗,

    out_path = '../out/'
    out_potery_path = '../out/potery.txt'
    len_limit = False
    max_seq_len = 50
    max_gen_len = 50 #诗的最大长度
