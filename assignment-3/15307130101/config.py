import argparse
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class Config(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")

        parser.add_argument('--data_path', default="data/")
        parser.add_argument('--pickle_path', default="tang.npz")
        parser.add_argument('--lr', type=float, default=1e-3, help='lr for weights')
        parser.add_argument('--use_gpu', default=False)
        parser.add_argument('--epoch', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--maxlen', type=int, default=125)
        parser.add_argument('--max_gen_len', type=int, default=200)
        parser.add_argument('--model_path', default="./model.pth")
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--print_freq', type=int, default=1)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.use_gpu = cuda
        self.device = device

