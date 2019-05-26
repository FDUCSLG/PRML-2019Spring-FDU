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
parser.add_argument("--learning_rate_decay", type=float, default=0.9)
parser.add_argument("--steps_per_decay", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch_num", type=int, default=10)
parser.add_argument("--regularization_rate", type=float, default=1e-4)
parser.add_argument("--validation_proportion", type=float, default=0.125)
parser.add_argument("--steps_per_validation", type=int, default=5)
parser.add_argument("--cross_times", type=int, default=5)
parser.add_argument("--ultilize_num", type=int, default=4)

args = parser.parse_args()

learning_rate = args.learning_rate 
regularization_rate = args.regularization_rate 
learning_rate_decay = args.learning_rate_decay 
steps_per_decay = args.steps_per_decay 
batch_size = args.batch_size 
epoch_num = args.epoch_num

validation_proportion= args.validation_proportion 
steps_per_validation = args.steps_per_validation

a = args.cross_times
survive_validation_num = args.ultilize_num

np.random.seed(2019)

if __name__ == '__main__':
    # output the hyerparameters
    hyerparameters_information = {}
    hyerparameters_information["learning_rate"] = learning_rate
    hyerparameters_information["learning_rate_decay"] = learning_rate_decay
    hyerparameters_information["steps per decay"] = steps_per_decay  
    hyerparameters_information["regularization_rate"] = regularization_rate
    hyerparameters_information["epoch_num"] = epoch_num
    hyerparameters_information["batch_size"] = batch_size
    hyerparameters_information["validation_proportion"] = validation_proportion
    hyerparameters_information["steps_per_validation"] = steps_per_validation
    hyerparameters_information["cross_validation_times"] = a
    hyerparameters_information["survive_validation_num"] = survive_validation_num
    utils.box_dict(hyerparameters_information, "hyerparameters_information")

    dataset_train, dataset_test = handout.get_text_classification_datasets()
    categories = dataset_train.target_names
     
    # training data and labels
    
    training_data = np.array(dataset_train.data)
    training_labels = np.array(dataset_train.target)

    clean_training_data = utils.clean_dataset(training_data)
    mapping_dict = utils.build_mapping_dict(clean_training_data)
    feature_vector = utils.data2vec(clean_training_data, mapping_dict)
    
    # build the model
    softmax_model = model.Softmax_CrossEntropy_model(class_num=len(categories),
                                                     feature_length=feature_vector.shape[1],
                                                     learning_rate=learning_rate,
                                                     regularization_rate=regularization_rate)

    feature_vector = np.array(feature_vector)
    training_labels = np.array(training_labels)

    # store back 
    special_feature_vector = feature_vector
    special_labels = training_labels
    
    num_feature_vector = len(special_feature_vector)
    num_training_feature_vector = int(num_feature_vector * (1 - validation_proportion))
    num_validating_feature_vector = (num_feature_vector - num_training_feature_vector)

    best_weight_record = []
    best_validation_accur_record = []

    index_select_all = np.arange(num_feature_vector)

    #cross_validation_times = int(1 / validation_proportion)
    cross_validation_times = a

    utils.box("Training Start")
    
    graph = plt.subplot(1,1,1)

    for times in range(cross_validation_times):
        softmax_model.global_initialize()
        softmax_model.set_learning_rate(learning_rate)
        # add a position for weight storing , reset the best score as 0
        best_weight_record.append([])
        best_validation_accur_record.append([])
        best_step = 0
        best_accur = 0
        best_loss = 10
        # shuffle op
        np.random.shuffle(index_select_all)
        special_feature_vector = special_feature_vector[index_select_all,:]
        special_labels = special_labels[index_select_all]

        # divide training set and validation set
        div_training_feature_vector = special_feature_vector[0:num_training_feature_vector]
        div_training_labels = special_labels[0:num_training_feature_vector]
        div_validating_feature_vector = special_feature_vector[num_training_feature_vector:]
        div_validating_labels = special_labels[num_training_feature_vector:]
        
        # start to train

        present_epoch = 0
        example_num = num_training_feature_vector
        pbar = tqdm(range(epoch_num))
        step = 0
        

        index_select = np.arange(num_training_feature_vector)
        dt_feature_vector = div_training_feature_vector
        dt_labels = div_training_labels
        
        validation_accuracy = []
        validation_step = []
        
        for present_epoch in pbar:
            i = 0
            np.random.shuffle(index_select)
        
            dt_feature_vector = dt_feature_vector[index_select, :]
            dt_labels = dt_labels[index_select]
        
            while i<example_num:
                step += 1
                batch_input = dt_feature_vector[i:i+batch_size]
                batch_labels = dt_labels[i:i+batch_size]
                i = i+batch_size
                
                softmax_model.forward(batch_input, batch_labels)
                softmax_model.optimize()
                
                if step %10 == 1:
                    pbar.set_description("cross validation times: %d | step: %d | loss: %.4f"%(times+1, step, softmax_model.latest_loss))
                
                if step % steps_per_validation == 1:
                    validation_accuracy_record = []
                    validation_loss_record = []
                    # run once on the validation set
                    for temp_i in range(num_validating_feature_vector):
                        v_batch_input = div_validating_feature_vector[temp_i: temp_i+1]
                        v_batch_labels = div_validating_labels[temp_i: temp_i+1]
                        softmax_model.forward(v_batch_input, v_batch_labels)
                        validation_accuracy_record.append(softmax_model.latest_accuracy)
                        validation_loss_record.append(softmax_model.latest_loss)
                    v_final_accuracy = np.mean(validation_accuracy_record)
                    v_final_loss = np.mean(validation_loss_record)

                    validation_step.append(step)
                    validation_accuracy.append(v_final_accuracy)

                    if (v_final_accuracy > best_accur) or (v_final_accuracy == best_accur and v_final_loss < best_loss):
                        best_accur = v_final_accuracy
                        best_loss = v_final_loss 
                        best_weight_record[times] = softmax_model.weight
                        best_validation_accur_record[times] = v_final_accuracy
                        best_step = step
                        
                
                if step % steps_per_decay == (steps_per_decay-1):
                    softmax_model.learning_rate = softmax_model.learning_rate * learning_rate_decay
        
        graph.plot(validation_step, validation_accuracy)
        print("best accuracy on validation set: %.4f | best step: %d"%(best_accur, best_step))
    
    plt.show()
    
    # select the survival weights
    best_validation_accur_record = np.array(best_validation_accur_record)
    best_weight_record = np.array(best_weight_record)
    select_index = (np.argsort(best_validation_accur_record))[-1*int(survive_validation_num):]
    best_weight_record = (best_weight_record[select_index])
    best_weight = np.mean(best_weight_record, 0)
    
    softmax_model.set_weight(best_weight)
    
    print("")
    # do testing
    test_batch_size = 1
    testing_data = (dataset_test.data)
    testing_labels = np.array((dataset_test.target))
    clean_testing_data = utils.clean_dataset(testing_data)
    feature_vector = utils.data2vec(clean_testing_data, mapping_dict)

    example_num = len(feature_vector)
    
    utils.box("Testting Process")
    accuracy_record = []
    top2_accuracy_record = []
    
    for i in (range(example_num)):
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
    

