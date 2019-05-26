import numpy as np
import matplotlib.pyplot as plt

f = open("adagrad_res.txt")
adagrad_list = []
for li in f.readlines():
    l = len(li)
    for i in range(l):
        if li[i] == ':' and li[i-1]=='s':
            s = li[i+1:i+5]
            adagrad_list.append(float(s))
            

adagrad = np.array(adagrad_list)

f = open("sgdres.txt")
sgd_list = []
for li in f.readlines():
    l = len(li)
    for i in range(l):
        if li[i] == ':' and li[i-1]=='s':
            s = li[i+1:i+5]
            sgd_list.append(float(s))

sgd = np.array(sgd_list)

f = open("adam_3.txt")
adam_list = []
for li in f.readlines():
    l = len(li)
    for i in range(l):
        if li[i] == ':' and li[i-1]=='s':
            s = li[i+1:i+5]
            adam_list.append(float(s))

adam = np.array(adam_list)


plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

y = adagrad.copy()
l = y.shape[0]
x = list(range(l))

plt.plot(x, y, label="adagrad")


y = sgd.copy()
l = y.shape[0]
x = list(range(l))

plt.plot(x, y, label="sgd")

y = adam.copy()
l = y.shape[0]
x = list(range(l))

plt.plot(x, y, label="adam")


plt.legend()
plt.show()