import os
os.sys.path.append('..')

import handout
import matplotlib.pyplot as plt
import argparse
import numpy as np

import re 
import string
import time
import utils
import model
from tqdm import tqdm


learning_rate = 0.1
regularization_rate = 0.00001
learning_rate_decay = 1.0
steps_per_decay = 200
#batch_size = 10
epoch_num = 10000
max_update_step = 2000
batch_size_list = [1,32, 64, 128, 2247]

np.random.seed(0)

def train_run(model, feature_vector, training_labels, batch_size, max_update_step, steps_per_decay):
    time.sleep(2)
    print("training process:")
    present_epoch = 0
    step = 0
    example_num = len(feature_vector)
    pbar = tqdm(total=max_update_step)
    step_record = []
    loss_record = []

    index_select = np.arange(len(feature_vector))
    flag = 0

    for present_epoch in range(epoch_num):
        i = 0
        np.random.shuffle(index_select)
        
        feature_vector = feature_vector[index_select, :]
        training_labels = training_labels[index_select]
        
        while i<example_num:
            step += 1
            batch_input = feature_vector[i:i+batch_size]
            batch_labels = training_labels[i:i+batch_size]
            i = i+batch_size
            
            softmax_model.forward(batch_input, batch_labels)

            softmax_model.optimize()
            
            if step %5 == 1:
                pbar.set_description("step: %d | loss: %.4f"%(step, softmax_model.latest_loss))
                step_record.append(step)
                loss_record.append(softmax_model.latest_loss)
            
            if step % steps_per_decay == 0:
                softmax_model.learning_rate = softmax_model.learning_rate * learning_rate_decay
            pbar.update(1)
            
            if step == max_update_step:
                flag = 1
                break
        if flag == 1:
            break
    pbar.close()
    return step_record, loss_record 



if __name__ == '__main__':
    dataset_train, dataset_test = handout.get_text_classification_datasets()
    categories = dataset_train.target_names
     
    # training data and labels
    training_data = (dataset_train.data)
    training_labels = np.array((dataset_train.target))
    
    clean_training_data = utils.clean_dataset(training_data)
    mapping_dict = utils.build_mapping_dict(clean_training_data)
    feature_vector = utils.data2vec(clean_training_data, mapping_dict)
    print(len(feature_vector[0]))
    
    # build model
    softmax_model = model.Softmax_CrossEntropy_model(class_num=len(categories),
                                                     feature_length=feature_vector.shape[1],
                                                     learning_rate=learning_rate,
                                                     regularization_rate=regularization_rate)
    
     
    #softmax_model.global_initialize()
    #softmax_model.set_learning_rate(learning_rate)
    
    graph1 = plt.subplot(1,2,1)
    graph2 = plt.subplot(1,2,2)
    for batch_size in batch_size_list:
        softmax_model.global_initialize()
        softmax_model.set_learning_rate(learning_rate)
        label = "batch_size: %d"%(batch_size)
        step_record, loss_record = train_run(softmax_model, feature_vector, training_labels, batch_size, max_update_step, steps_per_decay)
        if batch_size == 1:
            graph1.plot(step_record, loss_record, label=label)
        else:
            graph1.plot(step_record, loss_record, label=label)
            graph2.plot(step_record, loss_record, label=label)
    
    graph1.set_xlabel("steps")
    graph1.set_ylabel("loss", rotation=0)
    graph2.set_xlabel("steps")
    graph2.set_ylabel("loss", rotation=0)

    graph1.legend(loc='best', prop={'size':14})
    graph2.legend(loc='best', prop={'size':14})
    plt.show()



