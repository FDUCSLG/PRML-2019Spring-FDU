import numpy as np
import matplotlib.pyplot as plt
import math
from gm1d_plot import gm1d_plot

def Square_root_Rule(data_num):
    """
        Square_root_Rule
        Params:
            data_num: int. number of sampled_data.
        Return:
            bin_number
    """
    return math.ceil(math.pow(data_num, 1/2))

def Struge_Rule(data_num):
    """
        Struge_Rule
        Params:
            data_num: int. number of sampled_data.
        Return:
            bin_number
    """
    return math.ceil(1 + 3.322 * math.log(data_num, 10))

def Doane_Rule(sampled_data, data_num):
    """
        Doane Rule
        Params:
            sampled_data: [array]. points data
            data_num: [int]. number of sampled_data.
        Return:
            bin_number
    """
    param_middle = (sampled_data - np.mean(sampled_data))
    temp1 = [math.pow(i, 3) for i in param_middle]
    temp2 = [math.pow(i, 2) for i in param_middle]
    temp3 = 6*(data_num - 1)/((data_num + 1)*(data_num + 3))
    param_b = np.sum(temp1)/math.pow(np.sum(temp2), 3/2)
    param_b = np.abs(param_b)
    param_c = math.pow(temp3, 1/2)
    Doane_bin_num = math.ceil(math.log(data_num, 2) + 1 + math.log(1 + (param_b/param_c), 2))
    return Doane_bin_num

def Scott_Rule(sampled_data, data_num):
    """
        Scott Rule
        Params:
            sampled_data: [array]. points data
            data_num: [int]. number of sampled_data.
        Return:
            bin_number
    """
    Scott_bin_width = (3.5 * np.std(sampled_data) / math.pow(data_num, 1/3))
    data_range = max(sampled_data) - min(sampled_data)
    Scott_bin_num = math.ceil(data_range/Scott_bin_width)
    return Scott_bin_num

def Freedman_Rule(sampled_data, data_num):
    """
        Freedman Rule
        Params:
            sampled_data: [array]. points data
            data_num: [int]. number of sampled_data.
        Return:
            bin_number
    """
    sequence_data = np.sort(sampled_data)
    length = len(sequence_data)
    Q1 = np.mean(sequence_data[0:int(length/2)])
    Q3 = np.mean(sequence_data[int(length/2)+(length%2):])
    IQR = (Q3-Q1)
    Freedman_bin_width = (2 * IQR / math.pow(data_num, 1/3))
    data_range = max(sampled_data) - min(sampled_data)
    Freedman_bin_num = math.ceil(data_range/Freedman_bin_width)
    return Freedman_bin_num

def SAS(sampled_data, n_min=1, n_max=50):
    """
        Shimazaki and Shinomoto's choice
        input : data
        output: bin number
    """
    x_max = max(sampled_data)
    x_min = min(sampled_data)
    N_MIN = n_min
    N_MAX = n_max
    N = range(N_MIN,N_MAX)
    N = np.array(N)
    D = (x_max-x_min)/N
    C = np.zeros(shape=(np.size(D),1))
    
    plt.figure()
    temp_graph = plt.subplot(1,1,1)
    for i in range(np.size(N)):
        edges = np.linspace(x_min,x_max,N[i]+1)
        ki = temp_graph.hist(sampled_data,edges)
        ki = ki[0]
        k = np.mean(ki)
        v = sum((ki-k)**2)/N[i]
        C[i] = (2*k-v)/((D[i])**2)
    
    temp_graph.cla()
    cmin = min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0])
    optD = D[idx]
    
    fig = plt.figure()
    plt.title("Shimazaki and Shinomoto's choice optimization process")
    plt.plot(D,C,'.b',optD,cmin,'*r')
    
    return N[idx]+1

def plot_subgraph(gm1d, sampled_data, bin_num, graph, graph_title=None):
    #gm1d.plot(graph=graph)
    gm1d_plot(gm1d, graph)
    graph.hist(sampled_data, normed=True, bins=bin_num)
    graph.set_title(graph_title)

def estimation_dif_rule_histogram(sampled_data, gm1d):
    data_num = len(sampled_data)
    sampled_data = np.array(sampled_data)
    # generate sub-graphs
    sub_graph = []
    for i in range(8):
        sub_graph.append(plt.subplot(2,4,i+1))
    
    # Square_root Rule
    Square_bin_num = Square_root_Rule(data_num)
    plot_subgraph(gm1d, sampled_data, Square_bin_num, sub_graph[0], "Square Rule:"+str(Square_bin_num))
    

    # Struge's Rule
    Struge_bin_num = Struge_Rule(data_num)
    plot_subgraph(gm1d, sampled_data, Struge_bin_num, sub_graph[1], "Struge Rule:"+str(Struge_bin_num))
    
    # Doane's Rule
    Doane_bin_num = Doane_Rule(sampled_data, data_num)
    plot_subgraph(gm1d, sampled_data, Doane_bin_num, sub_graph[2], "Doane Rule:"+str(Doane_bin_num))
    
    # Rice's Rule
    Rice_bin_num = int(math.pow(data_num, 1/3) * 2)
    plot_subgraph(gm1d, sampled_data, Rice_bin_num, sub_graph[3], "Rice Rule:"+str(Rice_bin_num))

    # Scott's Rule
    Scott_bin_num = Scott_Rule(sampled_data, data_num)
    plot_subgraph(gm1d, sampled_data, Scott_bin_num, sub_graph[4], "Scott Rule:"+str(Scott_bin_num))

    # Freedman Rule
    Freedman_bin_num = Freedman_Rule(sampled_data, data_num)
    plot_subgraph(gm1d, sampled_data, Freedman_bin_num, sub_graph[5], "Freedman Rule:"+str(Freedman_bin_num))

    # Shimazaki and Shinomoto's choice
    SAS_bin_num = SAS(sampled_data, n_min=1, n_max=75)
    plot_subgraph(gm1d, sampled_data, SAS_bin_num, sub_graph[6], "Shimazaki and Shinomoto Rule:"+str(SAS_bin_num))

    plt.show()
