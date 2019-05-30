import matplotlib.pyplot as plt
from data.preposess import *

def compare_optim():
    adam=load('.record/loss/record single with trick.txt')
    adam_001=load('.record/loss/record single loss adam 0.01.txt')
    mom=load('.record/loss/record single loss mom.txt')
    SGD=load('.record/loss/record single loss SGD.txt')
    rms=load('.record/loss/record single loss rms.txt')
    x=range(40)
    plt.plot(x,adam,label = "Adam 0.001", color='blue')
    plt.plot(x,adam_001,label = "Adam 0.01", color='blue', linestyle="--")
    plt.plot(x,mom,label = "Momentum", color='red')
    plt.plot(x,SGD,label = "SGD", color='green')
    plt.plot(x,rms,label = "RMSprop", color='orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('optimizor vs loss')
    plt.legend()
    plt.savefig("figs/optim.png", dpi=120)
    plt.show()

# compare_optim()

def compare_pretrain():
    adam=load('.record/loss/record single without pre-train.txt')
    mom=load('.record/loss/record single pre-train fixed.txt')
    SGD=load('.record/loss/record single with trick.txt')
    x=range(40)
    plt.plot(x,adam,label = "without pre-train", color='blue')
    plt.plot(x,mom,label = "fixed pre-train", color='red')
    plt.plot(x,SGD,label = "normal", color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('pre-train vs loss')
    plt.legend()
    plt.savefig("figs/pretrain.png", dpi=120)
    plt.show()

# compare_pretrain()

def compare_trick():
    adam=load('.record/loss/record single with trick.txt')
    mom=load('.record/loss/record single without trick.txt')
    x=range(40)
    plt.plot(x,adam,label = "without trick", color='green')
    plt.plot(x,mom,label = "with trick", color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('trick vs loss')
    plt.legend()
    plt.savefig("figs/trick.png", dpi=120)
    plt.show()

compare_optim()