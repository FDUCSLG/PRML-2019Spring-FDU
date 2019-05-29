import argparse
from data_processing import *
from CNN_Text import myCNNText
from LSTM_Text import myRNNText
from fastNLP import CrossEntropyLoss
from fastNLP import Adam
from fastNLP import AccuracyMetric
from fastNLP import Trainer, Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', choices=['train', 'test'], help='train or test')
parser.add_argument('--model', choices=['CNN', 'RNN'], help='CNN or RNN')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--embed_size', default=50, type=int, metavar='N', help='embed size')
parser.add_argument('--model_path', type=str, help='filename of saved model')
args = parser.parse_args()

if args.mode == 'train':
    train_data, test_data, vocab = get_fastnlp_dataset()
    weight = get_pretrained_weight(vocab, args.embed_size)
    if args.model == 'CNN':
        model = myCNNText(embed_num=len(vocab), embed_dim=args.embed_size, num_classes=8, padding=2, dropout=0.1, pre_weight=weight)
    else:
        model = myRNNText(embed_num=len(vocab), embed_dim=args.embed_size, num_classes=8, hidden_dim=128, num_layer=1, bidirectional=False, pre_weight=weight)

    trainer = Trainer(train_data=train_data, model=model,
                      loss=CrossEntropyLoss(pred='pred', target='target'),
                      metrics=AccuracyMetric(),
                      n_epochs=args.epochs,
                      batch_size=args.batch_size,
                      print_every=-1,
                      validate_every=-1,
                      dev_data=test_data,
                      save_path='./model',
                      optimizer=Adam(lr=args.lr, weight_decay=0),
                      check_code_level=-1,
                      metric_key='acc',
                      use_tqdm=False,
                      )
    trainer.train()

if args.mode == 'test':
    train_data, test_data, vocab = get_fastnlp_dataset()
    weight = get_pretrained_weight(vocab, args.embed_size)
    if args.model == 'CNN':
        model = myCNNText(embed_num=len(vocab), embed_dim=args.embed_size, num_classes=8, padding=2, dropout=0.1, pre_weight=weight)
    else:
        model = myRNNText(embed_num=len(vocab), embed_dim=args.embed_size, num_classes=8, hidden_dim=128, num_layer=1, bidirectional=False, pre_weight=weight)

    model = load_model(model, './model/' + args.model_path)

    tester = Tester(test_data, model, metrics=AccuracyMetric())
    eval_results = tester.test()
    print(eval_results)
