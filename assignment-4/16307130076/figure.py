import matplotlib.pyplot as plt
import csv
import argparse

def export_figure(datapath,ylabel):
    csvfile = open(datapath, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
	    y.append((row[2]))
	    x.append((row[1]))

    plt.plot(x, y)

    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--ylabel", type=str, default="Score")
    arg = parser.parse_args()
    export_figure(arg.datapath,arg.ylabel)
