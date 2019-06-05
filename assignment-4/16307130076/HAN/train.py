import os
import sys
sys.path.append('.')
from HAN.model import HierNet
from HAN.dataset import HANDataset
from utils import get_evaluation
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import shutil
from tensorboardX import SummaryWriter
import pandas as pd
import csv
from fetch_dataset import fetch_dataset, get_max_lengths



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-n", "--num_epochs", type=int, default=100)
    parser.add_argument("-l", "--lr", type=float, default=0.1)
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["20newsgroups", "ag_news", "amazon_review_full", "amazon_review_polarity", "dbpedia", "sougou_news", "yahoo_answers", "yelp_review_full", "yelp_review_polarity"], default="20newsgroups")
    parser.add_argument("--dict_path", type=str,
                        default="./data/glove.6B.50d.txt")
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0)
    parser.add_argument("-w", "--es_patience", type=int, default=5)
    parser.add_argument("-v", "--log_path", type=str,
                        default="tensorboard/han")
    args = parser.parse_args()
    return args


def train(opt):
    dataset_train, dataset_test = fetch_dataset(opt.dataset)
    dict = pd.read_csv(filepath_or_buffer=opt.dict_path,
                       header=None, sep=" ", quoting=csv.QUOTE_NONE).values
    num_classes = dataset_train.num_classes
    word_length, sent_length = get_max_lengths(dataset_train)
    dataset_train = HANDataset(dataset_train, dict, sent_length, word_length)
    dataset_test = HANDataset(dataset_test, dict, sent_length, word_length)
    train_generator = DataLoader(
        dataset_train, batch_size=opt.batch_size, shuffle=True)
    test_generator = DataLoader(
        dataset_test, batch_size=opt.batch_size, shuffle=False)

    model = HierNet(dict, opt.batch_size, opt.word_hidden_size,
                    opt.sent_hidden_size, num_classes)

    log_path = "{}_{}".format(opt.log_path, opt.dataset)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(train_generator)

    for epoch in range(opt.num_epochs):
        for iter, batch in enumerate(train_generator):
            feature, label = batch
            label = label.long()
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden(len(label))
            pred = model(feature)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), pred.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            print("Epoch: %d/%d, Iteration: %d/%d, lr: %.6f, loss: %.6f, acc: %.6f" % (epoch+1, opt.num_epochs,
                                                                                       iter+1, num_iter_per_epoch, optimizer.param_groups[0]['lr'], loss.item(), training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch *
                              num_iter_per_epoch + iter)
            writer.add_scalar(
                'Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for batch in test_generator:
            with torch.no_grad():
                te_feature, te_label = batch
                te_label = te_label.long()
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                model._init_hidden(len(te_label))
                te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss.item() * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())

        te_loss = sum(loss_ls) / len(dataset_test)
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=[
                                      "accuracy", "confusion_matrix"])
        print("Epoch: %d/%d, loss: %.6f, acc: %.6f,confusion matrix: \n%s" %
              (epoch + 1, opt.num_epochs, te_loss, test_metrics["accuracy"], test_metrics["confusion_matrix"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()

        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "./weights/han_%s_%d.pth" %
                       (opt.dataset, epoch))
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch %d. The lowest loss achieved is %.6f at epoch %d" % (
                epoch, best_loss, best_epoch))
            break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
