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

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1.0)
parser.add_argument("--learning_rate_decay", type=float, default=0.95)
parser.add_argument("--steps_per_decay", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch_num", type=int, default=100)
parser.add_argument("--regularization_rate", type=float, default=1e-4)
parser.add_argument("--auto_terminate", type=str, default="False",choices=['True','False'])
parser.add_argument("--observe_loss_sequence_length", type=int, default=20)
parser.add_argument("--terminate_threshold", type=float, default=0.01)

args = parser.parse_args()

learning_rate = args.learning_rate
regularization_rate = args.regularization_rate 
learning_rate_decay = args.learning_rate_decay
steps_per_decay = args.steps_per_decay
batch_size = args.batch_size
max_epoch_num = args.max_epoch_num

if args.auto_terminate == "True":
    auto_terminate = True
else:
    auto_terminate = False 

observe_dif_times = args.observe_loss_sequence_length
terminate_threshold = args.terminate_threshold


np.random.seed(2019)


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
    present_epoch = 0
    example_num = len(feature_vector)
    step = 0
    
    # initial auto_terminate
    import queue
    loss_dif_queue = queue.Queue()
    for i in range(observe_dif_times):
        loss_dif_queue.put(100.0)
    abs_loss_dif_mean = 100.0
    
    # loss record variables
    even_step_loss = 0
    odd_step_loss = 0

    # break flag
    break_flag = False
    
    # output the hyerparameters
    hyerparameters_information = {}
    hyerparameters_information["learning_rate"] = learning_rate
    hyerparameters_information["learning_rate_decay"] = learning_rate_decay 
    hyerparameters_information["steps per decay"] = steps_per_decay
    hyerparameters_information["regularization_rate"] = regularization_rate
    hyerparameters_information["max_epoch_num"] = max_epoch_num
    hyerparameters_information["batch_size"] = batch_size
    hyerparameters_information["auto_terminate"] = auto_terminate
    if auto_terminate == True:
        hyerparameters_information["observation of loss num"] = observe_dif_times
        hyerparameters_information["terminate_threshold"] = terminate_threshold
    print("")
    utils.box_dict(hyerparameters_information, "hyerparameters_information")

    # start training
    utils.box("Training Process")

    index_select = np.arange(len(feature_vector))
    pbar = tqdm(range(max_epoch_num))
    
    for present_epoch in pbar:
        i = 0
        # shuffle data
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
            
            if step %10 == 1:
                pbar.set_description("step: %d | loss: %.4f"%(step, softmax_model.latest_loss)) 
            
            if step % steps_per_decay == 0:
                softmax_model.learning_rate = softmax_model.learning_rate * learning_rate_decay
            
            # auto terminate judgement
            if auto_terminate == True:
                if step % 2 == 1:
                    odd_step_loss = softmax_model.latest_loss
                else:
                    even_step_loss = softmax_model.latest_loss
                    dif = np.abs(odd_step_loss - even_step_loss)
                    latest_loss = loss_dif_queue.get()
                    abs_loss_dif_mean = abs_loss_dif_mean - latest_loss/observe_dif_times + dif/observe_dif_times
                    loss_dif_queue.put(dif)
                    
                    # judge whether to stop
                    if abs_loss_dif_mean < terminate_threshold:
                        break_flag = True
                        break
        
        if break_flag == True:
            print('\n')
            inf_record = ("loss-differences mean of the recent %d steps is : %f"%(observe_dif_times*2, abs_loss_dif_mean))
            utils.box("Auto Terminate Condition Meets", other="  =>> "+inf_record)
            print(" ")
            break

    # do testing
    test_batch_size = 1
    testing_data = (dataset_test.data)
    testing_labels = np.array((dataset_test.target))
    clean_testing_data = utils.clean_dataset(testing_data)
    feature_vector = utils.data2vec(clean_testing_data, mapping_dict)
    
    example_num = len(feature_vector)
    
    utils.box("Testing Result")
    accuracy_record = []
    top2_accuracy_record = []
    for i in range(example_num):
        batch_input = feature_vector[i:i+1]
        batch_labels = testing_labels[i:i+1]
        softmax_model.forward(batch_input, batch_labels)
        accuracy_record.append(softmax_model.latest_accuracy)
        if softmax_model.latest_accuracy == 0:
            top2_index = np.argsort(softmax_model.predictions[0])[-2]
            if top2_index == testing_labels[i:i+1][0]:
                top2_accuracy_record.append(1)
            else:
                top2_accuracy_record.append(0)
        else:
            top2_accuracy_record.append(1)    
    
    print("accuracy mean on test set: %.6f"%(np.mean(accuracy_record)))
    print("top2 accuracy on test set: %.6f"%(np.mean(top2_accuracy_record)))


