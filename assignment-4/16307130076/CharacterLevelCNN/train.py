
from tensorboardX import SummaryWriter
import sys
sys.path.append('.')
import os   
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from fetch_dataset import fetch_dataset
from CharacterLevelCNN.dataset import CharacterDataset
from utils import get_evaluation
from CharacterLevelCNN.model import CharacterLevelCNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-n", "--num_epochs", type=int, default=20)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["20newsgroups","ag_news","amazon_review_full","amazon_review_polarity","dbpedia","sougou_news","yahoo_answers","yelp_review_full","yelp_review_polarity"], default="20newsgroups")
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0)
    parser.add_argument("-w", "--es_patience", type=int, default=3)
    parser.add_argument("-v", "--log_path", type=str,
                        default="tensorboard/char-cnn")
    args = parser.parse_args()
    return args


def train(opt):
    dataset_train, dataset_test = fetch_dataset(opt.dataset)
    dataset_train = CharacterDataset(dataset_train, opt.max_length)
    dataset_test = CharacterDataset(dataset_test, opt.max_length)
    train_generator = DataLoader(
        dataset_train, batch_size=opt.batch_size, shuffle=True)
    test_generator = DataLoader(
        dataset_test, batch_size=opt.batch_size, shuffle=False)

    if opt.feature == "small":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=dataset_train.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024,init_std=0.05)

    elif opt.feature == "large":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=dataset_test.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048, init_std=0.02)
    else:
        sys.exit("Invalid feature mode!")

    log_path = "{}_{}_{}".format(opt.log_path, opt.feature, opt.dataset)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
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
              (epoch + 1, opt.num_epochs, te_loss, test_metrics["accuracy"],test_metrics["confusion_matrix"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()

        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "./weights/char-cnn_%s_%s_%d.pth" %
                       (opt.dataset, opt.feature, epoch))
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch %d. The lowest loss achieved is %.6f at epoch %d" % (
                epoch, best_loss, best_epoch))
            break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
