import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..fetch_dataset import fetch_dataset
from model import CharacterLevelCNN
from dataset import CharacterDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--alphabet)