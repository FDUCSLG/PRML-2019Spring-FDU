from data_processing import *
from model_batched import *
from train import *
from generate_poetry import *
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', choices=['train', 'generate'], help='train or generate')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--embed_size', default=256, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden_size', default=256, type=int, metavar='N', help='lstm hidden size')
parser.add_argument('--start_word', type=str, default="<START>", help='start word of the poem')
parser.add_argument('--max_length', type=int, default=95, metavar='N', help='max length of the poem')
args = parser.parse_args()

if args.mode == 'train':
    train(epochNum=args.epochs, batch=args.batch_size, lr=args.lr, embed_size=args.embed_size, hidden_size=args.hidden_size)

if args.mode == 'generate':
    print(generate(startWord=args.start_word, max_length=args.max_length))

