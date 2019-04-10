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


regularization_rate = 0.00001
learning_rate_decay = 1.0
steps_per_decay = 200
batch_size = 64
epoch_num = 10000
max_update_step = 5000

learning_rate_list = [10, 1.0, 0.1, 0.01, 0.001, 0.0001]
#learning_rate_list = [80, 10, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]

np.random.seed(2019)

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
                                                     regularization_rate=regularization_rate)
    
     
    #softmax_model.global_initialize()
    #softmax_model.set_learning_rate(learning_rate)
    
    graph1 = plt.subplot(1,2,1)
    graph2 = plt.subplot(1,2,2)
    for learning_rate in learning_rate_list:
        softmax_model.global_initialize()
        softmax_model.set_learning_rate(learning_rate)
        label = "learning_rate: %f"%(learning_rate)
        step_record, loss_record = train_run(softmax_model, feature_vector, training_labels, batch_size, max_update_step, steps_per_decay)
        graph1.plot(step_record, loss_record, label=label)
        graph2.plot(step_record[20:], loss_record[20:], label=label)
    
    graph1.set_xlabel("steps")
    graph1.set_ylabel("loss", rotation=0)
    graph2.set_xlabel("steps")
    graph2.set_ylabel("loss", rotation=0)

    graph1.legend(loc='best', prop={'size':14})
    graph2.legend(loc='best', prop={'size':14})

    plt.show()

    # do testing
    test_batch_size = 1
    testing_data = (dataset_test.data)
    testing_labels = np.array((dataset_test.target))
    clean_testing_data = utils.clean_dataset(testing_data)
    feature_vector = utils.data2vec(clean_testing_data, mapping_dict)

    example_num = len(feature_vector)
    
    """
    print("testting process:")
    accuracy_record = []
    top2_accuracy_record = []
    for i in tqdm(range(example_num)):
        batch_input = feature_vector[i:i+1]
        batch_labels = testing_labels[i:i+1]
        softmax_model.forward(batch_input, batch_labels)
        #print(softmax_model.latest_loss)
        #print(softmax_model.latest_accuracy)
        accuracy_record.append(softmax_model.latest_accuracy)
        if softmax_model.latest_accuracy == 0:
            top2_index = np.argsort(softmax_model.predictions[0])[-2]
            if top2_index == testing_labels[i:i+1][0]:
                #print(top2_index)
                #print(testing_labels[i:i+1][0])
                top2_accuracy_record.append(1)
            else:
                top2_accuracy_record.append(0)
        else:
            top2_accuracy_record.append(1)    

    print("accuracy mean on test set: %.6f"%(np.mean(accuracy_record)))
    print("top2 accuracy on test set: %.6f"%(np.mean(top2_accuracy_record)))
    """



