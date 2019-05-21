import numpy as np
import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import time

class Least_Square_model(object):
    def __init__(self,input_data=None, labels=None, **unused_params):
        self.input_data = input_data
        self.labels = labels
        self.weight = None
        self.accuracy = None
    
    # train the best weight
    def run(self):
        self.input_data = np.array(self.input_data)
        self.labels = np.array(self.labels)
        component_1 = np.linalg.inv(np.matmul(self.input_data.transpose(), self.input_data))
        component_1 = np.dot(component_1, self.input_data.transpose())
        weight = np.dot(component_1, self.labels)
        self.weight = weight
        return weight
    
    def plot(self, graph):
        plot_x = np.linspace(-1.5, 1.5, 100)
        plot_y = (-1) * (self.weight[0] + self.weight[1] * plot_x) / self.weight[2]
        graph.scatter(self.input_data[:,1], self.input_data[:,2], c=self.labels)
        plot_line = str(self.weight[1]) + " * X + " + str(self.weight[2]) + " * Y + " + str(self.weight[0])
        plot_line_label = "seperate_line: (%.4f)*x1+(%.4f)*x2+(%.4f)=0"%(self.weight[1], self.weight[2], self.weight[0])
        self.accuracy = self.accuracy_cal()
        accuracy_label = "accuracy: %.4f"%(self.accuracy)
        line_label = plot_line_label + "\n" + accuracy_label
        graph.set_title("Least-Square model", fontsize=30)
        graph.plot(plot_x, plot_y, label=line_label)


    def accuracy_cal(self):
        prediction = self.weight[0] + self.weight[1] * self.input_data[:,1] + self.weight[2] * self.input_data[:, 2]
        prediction = [1 if i > 0 else -1 for i in prediction]
        accuracy_val = float((prediction == self.labels).mean())
        return accuracy_val

class Perception_model(object):
    def __init__(self, input_data=None, labels=None, learning_rate=0.02, max_epoch=100):
        self.input_data = input_data
        self.labels = labels
        self.weight = None
        self.learning_rate = learning_rate 
        self.max_epoch = max_epoch
        self.accuracy = None
    # getter
    def get_weight(self):
        return weight
    
    def get_accuracy(self):
        return self.accuracy

    # setter
    def set_input_data(self, input_data):
        self.input_data = input_data

    def set_labels(self, labels):
        self.labels = labels
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    
    # train the model 
    def run(self):
        self.input_data = np.array(self.input_data)
        self.labels = np.array(self.labels)
        self.weight = np.ones(self.input_data.shape[1])
        epoch = 0
        while True:
            epoch += 1
            if epoch > self.max_epoch:
                print("epoch: " + str(epoch))
                break
            
            # start an epoch training
            mislabel_num = 0
            for i in range(self.input_data.shape[0]):
                if np.dot(self.weight, np.transpose(self.input_data[i]))*self.labels[i] < 0:
                    mislabel_num += 1
                    self.weight  = self.weight + self.learning_rate * self.input_data[i] * self.labels[i]
                else:
                    pass
                
            if mislabel_num == 0:
                break
        return self.weight
    
    def plot(self, graph):
        plot_x = np.linspace(-1.5, 1.5, 100)
        plot_y = (-1) * (self.weight[0] + self.weight[1] * plot_x) / self.weight[2]
        graph.scatter(self.input_data[:,1], self.input_data[:,2], c=self.labels)
        plot_line = str(self.weight[1]) + " * X + " + str(self.weight[2]) + " * Y + " + str(self.weight[0])
        plot_line_label = "seperate_line: (%.4f)*x1+(%.4f)*x2+(%.4f)=0"%(self.weight[1], self.weight[2], self.weight[0])
        self.accuracy = self.accuracy_cal()
        accuracy_label = "accuracy: %.4f"%(self.accuracy)
        line_label = plot_line_label + "\n" + accuracy_label
        graph.set_title("Perception model", fontsize=30)
        graph.plot(plot_x, plot_y, label=line_label)
         
    def accuracy_cal(self):
        prediction = self.weight[0] + self.weight[1] * self.input_data[:,1] + self.weight[2] * self.input_data[:, 2]
        prediction = [1 if i > 0 else -1 for i in prediction]
        accuracy_val = float((prediction == self.labels).mean())
        return accuracy_val 

class Softmax_CrossEntropy_model(object):
    def __init__(self, class_num, feature_length, learning_rate=0.002, 
                 regularization_rate=0.0):
        # props and trainable params
        self.class_num = class_num
        self.feature_length = feature_length
        #self.weight = np.ones((class_num, feature_length)) * 0.05
        self.weight = 0.01 * np.random.randn(class_num, feature_length)
        # high parameters
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.step = 0
        self.batch_size = 128

        # record of the latest information
        self.input_data = []
        self.labels = []
        self.predictions = []

        self.latest_loss = -1
        self.latest_accuracy = -1
    
    # getter
    def get_predictions(self):
        return self.predictions
    
    def get_weight(self):
        return self.weight 
    
    def get_loss(self):
        return self.latest_loss 

    # setter
    def set_learning_rate(self, rate):
        self.learning_rate = rate 

    def set_regularization_rate(self, rate):
        self.regularization_rate = rate 

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size 
    
    def set_weight(self, weight):
        self.weight = weight
    
    def global_initialize(self):
        self.weight = 0.01 * np.random.randn(self.class_num, self.feature_length)

    # layer
    def softmax_layer(self, input_data):
        input_data = input_data - np.max(input_data, axis=1).reshape(-1,1)
        numerator = np.exp(input_data)
        denominator = np.expand_dims(np.sum(numerator, axis=1), axis=1)
        softmax_output = numerator / denominator
        return softmax_output

    def cross_entropy_loss(self, predictions, labels):
        epsilon = 1e-8
        # the labels need to be one-hot format
        float_labels = np.array(labels).astype(np.float32)
        cross_entropy_loss_val = (-1.0) * float_labels * np.log(predictions + epsilon)
        cross_entropy_loss_val = np.mean(np.sum(cross_entropy_loss_val, axis=1))
        return cross_entropy_loss_val
    
    def accuracy_cal(self, predictions, labels):
        example_num = len(labels)
        right_num = np.sum(np.argmax(predictions, axis=1)==np.argmax(labels))
        return (1.0) * right_num / example_num

    def forward(self, feature, labels):
        self.input_data = feature 
        self.labels = np.eye(self.class_num)[labels]

        # fully connected layer
        fc_output = np.dot(feature, np.transpose(self.weight))
        # softmax layer
        softmax_output = self.softmax_layer(fc_output)
        # cross_entropy_loss
        cross_entropy_loss_val = self.cross_entropy_loss(softmax_output, self.labels)
        accuracy_val = self.accuracy_cal(softmax_output, self.labels)
        
        # record the latest information
        self.predictions = softmax_output
        self.latest_loss = cross_entropy_loss_val 
        self.latest_accuracy = accuracy_val

    def optimize(self):
        # optimize the weight
        gradient = np.zeros(self.weight.shape)
        gradient = np.dot(np.transpose(self.input_data), self.predictions - self.labels)
        gradient = np.transpose(gradient / len(self.input_data))
        
        reg_bp = self.weight.copy()
        reg_bp[:,0] = 0
        #self.weight = self.weight - self.learning_rate * gradient
        self.weight = self.weight - self.learning_rate * gradient - self.regularization_rate * reg_bp * reg_bp







