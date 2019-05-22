
class Config(object):
    dataset = 'large'
    # dataset = 'small'

    epoch = 50
    embedding_size = 128
    hidden_size = 512

    def __init__(self):
        if self.dataset == 'small':
            self.data_path = '../handout/tangshi.txt'
        else:
            self.data_path = 'tang.npz'

    # output_dir = './'
    # do_train = True
    #
    # pickle_path = 'tang.npz'
    # author = None  # 只学习某位作者的诗歌
    # constrain = None  # 长度限制
    # category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    # lr = 1e-3
    # weight_decay = 1e-4
    # use_gpu = True
    # epoch = 20
    # batch_size = 128
    # maxlen = 125
    # plot_every = 20
    # # use_env = True
    # env = 'poetry'  # visdom env
    # max_gen_len = 200
    # debug_file = '/tmp/debugp'
    # model_path = None
    # prefix_words = '细雨鱼儿出,微风燕子斜。'
    # start_words = '闲云潭影日悠悠'
    # acrostic = False  # 是否是藏头诗
    # model_prefix = 'checkpoints/tang'
