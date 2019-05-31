import sys
import os
import pickle
import argparse
import logging
import fitlog
sys.path.append('../')
import torch
from gensim.models import word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.decomposition import TruncatedSVD

from fastNLP.models import CNNText,STSeqCls
from fastNLP import Trainer,Tester,Batch,Const
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core import optimizer as fastnlp_optim
from fastNLP.io import EmbedLoader
from fastNLP.core.callback import FitlogCallback,EarlyStopCallback
from copy import deepcopy


from dataset import TextData
from model import LSTMText,MyCNNText

fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file (__file__) # record your hyperparameters


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('PRML assignment4')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain the model using pretrain_model')
    # parser.add_argument('--predict', action='store_true',
    #                     help='predict the answers for test set with trained model')
    parser.add_argument('--cuda', action='store_true',
                        help='using cuda')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', choices=['Adam', 'None','Adagrad','Adadelta'],default='None',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout', type=float, default=0.3,
                                help='dropout rate')
    train_settings.add_argument('--num_layers', type=float, default=2,
                                help='dropout rate')                                
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--patience', type=int, default=100,
                                help='patience of early stop')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--model', choices=['CNNText', 'MyCNNText','StarTransformer','LSTMText','Bert'], default='CNNText',
                                help='choose the algorithm to use')
    model_settings.add_argument('--pretrain_model', choices=['word2vec','glove','None','glove2wv'], default='None',
                                help='choose the pre_train model to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--min_count', type=int, default=10,
                                help='min num of the words in vocabulary')
    model_settings.add_argument('--max_seq_len', type=int, default=500,
                                help='max length of passage')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--data_src', default='20news',
                               help='the data source that will be loaded in prepare part.')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--vocab_data', default='vocab.data',
                               help='the file that stored the vocab(and the file to save)')
    path_settings.add_argument('--model_suffix', default='default',
                               help='the dir to store models')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--prepare_dir', default='../data/prepare/',
                               help='the dir to store the prepared data such as word2vec')
    path_settings.add_argument('--log_path',default='./run_records.log',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def prepare(args):
    print('Checking the data files...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.prepare_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # load dataset
    data = TextData(data_src=args.data_src,min_count=args.min_count,seq_limit=args.max_seq_len)
    data.fetch_data()

    print('Saving vocab(TextData)...')
    with open(os.path.join(args.vocab_dir, args.vocab_data), 'wb') as fout:
        pickle.dump(data, fout)

    print('Done with preparing!')

def word2vec_pretrain(args,sentences):
    print("Training Word2Vec...")
    word2vec_model = word2vec.Word2Vec( sentences.content,size=300,
                                        min_count=10,workers=8,iter=15)
    w2v_dict = {}
    print("Building word2vec embedding...")
    for word in word2vec_model.wv.vocab:
        w2v_dict[word] = word2vec_model[word]
    with open(os.path.join(args.prepare_dir,'w2v_dic.pkl'),'wb') as fout:
        pickle.dump(w2v_dict,fout)
    fout.close()
    print(word2vec_model)
    word2vec_model.wv.save_word2vec_format(
        os.path.join(args.prepare_dir,'w2v_model.txt'),binary=False)
    print("Word2Vec preparation done.")
    return os.path.join(args.prepare_dir,'w2v_model.txt')

def glove2wv_pretrain(args,sentences):
    assert(args.pretrain_model == 'glove2wv')
    print("Training Glove-Word2Vec...")
    glove_model_path = os.path.join(args.prepare_dir,'glove_model.txt')
    tmp_file = get_tmpfile("glove2wv_model.txt")
    _ = glove2word2vec(glove_model_path, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    model.save_word2vec_format(
            os.path.join(args.prepare_dir,'glove2wv_model.txt'),binary=False)
    print("Glvoe_Word2Vec preparation done.")
    return os.path.join(args.prepare_dir,'glove2wv_model.txt')

def pretrain_embedding(args):
    if not os.path.exists(args.prepare_dir):
        os.makedirs(args.prepare_dir)
    text_data = TextData()
    with open(os.path.join(args.vocab_dir, args.vocab_data), 'rb') as fin:
        text_data = pickle.load(fin)
    
    if args.pretrain_model == 'word2vec':
        w2v_model_path = word2vec_pretrain(args,text_data.train_set['text'])
        print("Load the glove model from {0}.".format(w2v_model_path))
        loader = EmbedLoader()
        vocab = text_data.vocab
        pretrained = loader.load_with_vocab(w2v_model_path,vocab)
        print(pretrained)
        save_path = os.path.join(args.prepare_dir,'w2v_embeds.pkl')
        with open(save_path,'wb') as fout:
            pickle.dump(pretrained,fout)
        print("Building word2vec done.Matrix saved in {0}".format(save_path))
        
    elif args.pretrain_model == 'glove':
        print("Using Glove...")
        glove_model_path = os.path.join(args.prepare_dir,'glove_model.txt')
        print("Load the glove model from {0}.".format(glove_model_path))
        loader = EmbedLoader()
        vocab = text_data.vocab
        pretrained = loader.load_with_vocab(glove_model_path,vocab)
        print(pretrained)
        save_path = os.path.join(args.prepare_dir,'glove_embeds.pkl')
        with open(save_path,'wb') as fout:
            pickle.dump(pretrained,fout)
        print("Building Glove done.Matrix saved in {0}".format(save_path))
    elif args.pretrain_model == 'glove2wv':
        print("Using Glove trained with word2vec...")
        glove_model_path = glove2wv_pretrain(args,text_data.train_set['text'])
        print("Load the glove model from {0}.".format(glove_model_path))
        loader = EmbedLoader()
        vocab = text_data.vocab
        pretrained = loader.load_with_vocab(glove_model_path,vocab)
        print(pretrained)
        save_path = os.path.join(args.prepare_dir,'glove2wv_embeds.pkl')
        with open(save_path,'wb') as fout:
            pickle.dump(pretrained,fout)
        print("Building Glove done.Matrix saved in {0}".format(save_path))
    
    else:
        print("No pretrain model will be used.")

def train(args):
    text_data = TextData()
    with open(os.path.join(args.vocab_dir, args.vocab_data), 'rb') as fin:
        text_data = pickle.load(fin)
    vocab_size = text_data.vocab_size
    class_num = text_data.class_num
    seq_len = text_data.max_seq_len
    print("(vocab_size,class_num,seq_len):({0},{1},{2})".format(vocab_size,class_num,seq_len))

    train_data = text_data.train_set
    test_dev_data = text_data.test_set
    train_data.set_input('words','seq_len')
    train_data.set_target('target')
    test_dev_data.set_input('words','seq_len')
    test_dev_data.set_target('target')
    test_data,dev_data = test_dev_data.split(0.2)

    test_data = test_dev_data
    init_embeds = None
    if args.pretrain_model == "None":
        print("No pretrained model with be used.")
        print("vocabsize:{0}".format(vocab_size))
        init_embeds = (vocab_size,args.embed_size)
    elif args.pretrain_model == "word2vec":
        embeds_path = os.path.join(args.prepare_dir,'w2v_embeds.pkl')
        print("Loading Word2Vec pretrained embedding from {0}.".format(embeds_path))
        with open(embeds_path,'rb') as fin:
            init_embeds = pickle.load(fin)
    elif args.pretrain_model == 'glove':
        embeds_path = os.path.join(args.prepare_dir,'glove_embeds.pkl')
        print("Loading Glove pretrained embedding from {0}.".format(embeds_path))
        with open(embeds_path,'rb') as fin:
            init_embeds = pickle.load(fin)
    elif args.pretrain_model == 'glove2wv':
        embeds_path = os.path.join(args.prepare_dir,'glove2wv_embeds.pkl')
        print("Loading Glove pretrained embedding from {0}.".format(embeds_path))
        with open(embeds_path,'rb') as fin:
            init_embeds = pickle.load(fin)
    else :
        init_embeds = (vocab_size,args.embed_size)


    if args.model == "CNNText":
        print("Using CNN Model.")
        model = CNNText(init_embeds, num_classes=class_num,
                        padding=2, dropout=args.dropout)
    elif args.model == "StarTransformer":
        print("Using StarTransformer Model.")
        model = STSeqCls(init_embeds,num_cls=class_num,
                        hidden_size=args.hidden_size)
    elif args.model == "MyCNNText":
        model = MyCNNText(  init_embeds=init_embeds, num_classes=class_num,
                            padding=2, dropout=args.dropout)
        print("Using user defined CNNText")
    elif args.model == "LSTMText":
        print("Using LSTM Model.")
        model = LSTMText(init_embeds=init_embeds,output_dim=class_num,
                        hidden_dim=args.hidden_size,num_layers=args.num_layers,
                        dropout=args.dropout)
    elif args.model == "Bert":
        print("Using Bert Model.")
    else:
        print("Using default model: CNNText.")
        model = CNNText((vocab_size,args.embed_size),num_classes=class_num,padding=2,dropout=0.1)
    print(model)
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = None

    print("train_size:{0} ; dev_size:{1} ; test_size:{2}".format
        ( train_data.get_length(),dev_data.get_length(),test_data.get_length() ) )
    
    if args.optim == "Adam":
        print("Using Adam as optimizer.")
        optimizer = fastnlp_optim.Adam(lr=0.001,weight_decay=args.weight_decay)
        if(args.model_suffix=="default"):
            args.model_suffix == args.optim
    else:
        print("No Optimizer will be used.")
        optimizer = None
    
    criterion = CrossEntropyLoss()
    metric = AccuracyMetric()
    model_save_path = os.path.join(args.model_dir,args.model,args.model_suffix)
    earlystop = EarlyStopCallback(args.patience)
    trainer = Trainer(train_data=train_data,model=model,save_path=model_save_path,
                        device = device,n_epochs=args.epochs,
                        optimizer=optimizer,dev_data=test_data,
                        loss=criterion,batch_size=args.batch_size,
                        metrics=metric,
                        callbacks=[FitlogCallback(test_data),earlystop])
    trainer.train()
    print("Train Done.")

    tester = Tester(data=test_data,model=model,metrics=metric,
                    batch_size=args.batch_size,device=device)
    tester.test()
    print("Test Done.")
    fitlog.finish()   


def run():
    args = parse_args()
    logger = logging.getLogger("lab4")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.pretrain:
        pretrain_embedding(args)
    if args.train:
        train(args)
if __name__ == "__main__":
    run()
