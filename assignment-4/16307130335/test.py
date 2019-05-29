from fastNLP import Tester, AccuracyMetric
import argparse
import torch
from train import handle_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="cnn_model.pth")
    parser.add_argument("--category", "-c", default=20, type=int)
    args = parser.parse_args()
    model = torch.load(args.file)
    train_data, test_data, dic_size = handle_data(args.category)
    t = Tester(test_data, model, metrics=AccuracyMetric(pred="pred",target='target'))
    print(args.file)
    t.test()


if __name__=="__main__":
    main()