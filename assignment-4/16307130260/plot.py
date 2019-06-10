import numpy as np
import matplotlib.pyplot as plt

filedic  = {"log1_cnn.log": "CNN", "log1_cnn_word2vec.log": "CNN+w2v"}
loss_step = 10
acc_step = 64
def get_data(filename):
    with open(filename, "r") as f:
        loss = []
        acc = []
        for line in f:
            if "train loss:" in line:
                loss.append(float(line.split()[6]))
            if " AccuracyMetric: acc=" in line:
                acc.append(float(line.split("=")[1]))
        print(len(loss))
        print(len(acc))
    return loss, acc

for filename, legend in filedic.items():
    loss, acc = get_data(filename)
    range(len(loss),loss_step)
    xl = range(loss_step, loss_step * len(loss) + loss_step, loss_step)
    plt.plot(xl, loss, label=legend)
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.cla()
for filename, legend in filedic.items():
    loss, acc = get_data(filename)
    range(len(acc),acc_step)
    xl = range(acc_step, acc_step * len(acc) + acc_step, acc_step)
    plt.plot(xl, acc, label=legend)
plt.xlabel("steps")
plt.ylabel("")
plt.legend()
plt.show()