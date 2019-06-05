from fastNLP.core.callback import Callback
import sys
sys.path.append("..")

class MyCallback(Callback):
    def on_backward_begin(self, loss):
        with open('.record/loss record cnn dev fast.txt','a') as f:
            f.write('%f '%loss)
            f.close()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        with open('.record/acc record cnn dev fast.txt','a') as f:
            f.write('%f '%float(eval_result['AccuracyMetric']['acc']))
            f.close()

import matplotlib.pyplot as plt

def write(addr,list):
    with open(addr,"w",encoding='UTF-8') as f:
        for num in list:
            f.write(str(num))
            f.write(' ')


def load(addr):
    with open(addr,"r",encoding='UTF-8') as f:
        line=f.readline()
        list_char=line.split()
        list_num=[]
        for num in list_char:
            list_num.append(float(num))
    return list_num


