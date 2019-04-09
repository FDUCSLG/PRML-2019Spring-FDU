from trainer import *
from logistic import *
from data_utils import *

import os
os.sys.path.append('../..')
from handout import get_text_classification_datasets

kwargs_global = {}
kwargs_global["learning_rate"] = 1e-1
kwargs_global["lr_decay"] = 1
kwargs_global["lr_decay_freq"] = 1
kwargs_global["batch_size"] = 2000
kwargs_global["num_epochs"] = 3000
kwargs_global["delta_loss"] = 0
kwargs_global["beta"] = 0.9
# args for output
kwargs_global["save_freq"] = 10
kwargs_global["check_freq"] = 30

def split_data(split_point=1600, data_set=None):
    x = np.array(data_set.data)
    y = np.array(data_set.target)
    num_classes = len(data_set.target_names)
    # divide into training set and validation set
    # shuffle data
    ind = np.arange(len(x))
    rng = np.random.RandomState(233)
    rng.shuffle(ind)
    x = x[ind]
    y = y[ind]
    # divide
    raw_data = {}
    raw_data['train_x'] = x[:split_point]
    raw_data['train_y'] = y[:split_point]
    raw_data['val_x'] = x[split_point:]
    raw_data['val_y'] = y[split_point:]
    return raw_data, num_classes

def plot_loss_acc(kwargs, info):
    best_val_acc, loss_history, train_acc_his, val_acc_his = info
    lr = kwargs['learning_rate']
    # save loss per save_freq steps
    save_freq = kwargs['save_freq']
    # check acc per check_freq epoches
    check_freq = kwargs['check_freq']

    # plot loss
    plt.subplots_adjust(hspace=0.4)
    
    plt.subplot(2,1,1)
    x_loss = (np.arange(len(loss_history)) + 1) * save_freq
    plt.plot(x_loss, loss_history)
    plt.title("Loss")
    plt.xlabel("Step")

    # plot acc
    plt.subplot(2,1,2)
    x_acc = np.arange(len(train_acc_his)) * check_freq
    plt.plot(x_acc, train_acc_his, '-o', label="training set")
    if len(val_acc_his) != 0:
        plt.plot(x_acc, val_acc_his, '-o', label="validation set")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.gcf().set_size_inches(6, 6)
    plt.show()

def get_test_acc(test_data, model):
    pred_test_y, _ = model.loss(test_data['x'])
    test_acc = np.sum(np.argmax(pred_test_y, axis=1) == np.argmax(
        test_data['y'], axis=1)) / len(test_data['x'])
    print("test acc: ")
    print(test_acc)

def choose_learning_rate(data, size_voca, num_classes):
    learning_rate = [ 1, 0.1, 0.01, 0.001 ]
    kwargs = kwargs_global.copy()

    # full batch need not to change anything

    # mini batch config
    kwargs["batch_size"] = 50
    kwargs["num_epochs"] = 400
    kwargs["save_freq"] = 20
    kwargs["check_freq"] = 4

    # SGD config
    # kwargs["batch_size"] = 1
    # kwargs["num_epochs"] = 100
    # kwargs["save_freq"] = 800
    # kwargs["check_freq"] = 4
    # train models and plot loss
    for lr in learning_rate:
        # train
        model = logistic_model(input_dims=size_voca, class_nums=num_classes, weight_scale=1e-2, reg=1e-4)
        kwargs['learning_rate'] = lr
        Train = trainer(model=model, data=data, kwargs=kwargs.copy())
        _, info = Train.train()
        # plot losses
        x_loss = (np.arange(len(info[1])) + 1) * kwargs['save_freq']
        plt.plot(x_loss, info[1], label=("lr: " + str(lr)))
    
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Step")
    plt.show()

def change_batch_size1(data, size_voca, num_classes):
    batch_sizes = [ 1, 20, 200, 2000 ]
    kwargs = kwargs_global.copy()
    kwargs['num_epochs'] = 30
    # train models and plot loss
    for bz in batch_sizes:
        # train
        model = logistic_model(input_dims=size_voca, class_nums=num_classes, weight_scale=1e-2, reg=1e-4)
        kwargs['batch_size'] = bz
        kwargs['save_freq'] = kwargs_global['save_freq'] // (1 + np.log(bz))
        Train = trainer(model=model, data=data, kwargs=kwargs.copy())
        _, info = Train.train()
        # plot losses
        x_loss = (np.arange(len(info[1])) + 1) * kwargs['save_freq'] * kwargs["batch_size"]
        plt.plot(x_loss, info[1], label=("batch size: " + str(bz)))

    plt.legend()
    plt.title("Loss")
    plt.xlabel("data num")
    plt.ylim(0, 2.5)
    plt.show()

def change_batch_size2(data, size_voca, num_classes):
    batch_sizes = [ 1, 20, 200, 2000 ]
    kwargs = kwargs_global.copy()
    epo = 2000
    
    # train models and plot loss
    for bz in batch_sizes:
        # train
        model = logistic_model(input_dims=size_voca, class_nums=num_classes, weight_scale=1e-2, reg=1e-4)
        ratio = kwargs_global['batch_size'] / bz
        kwargs['batch_size'] = bz
        kwargs['num_epochs'] = epo / ratio
        Train = trainer(model=model, data=data, kwargs=kwargs.copy())
        _, info = Train.train()
        # plot losses
        x_loss = (np.arange(len(info[1])) + 1) * kwargs['save_freq']
        plt.plot(x_loss, info[1], label=("batch size: " + str(bz)))

    plt.legend()
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylim(0, 2.5)
    plt.show()

def training_best_model(data, size_voca, num_classes, test_data, kwargs, learning_rate):
    best_val_acc = 0
    best_model = None
    best_kwargs = None
    best_info = None
    # train models and plot loss
    for lr in learning_rate:
        # train
        model = logistic_model(input_dims=size_voca, class_nums=num_classes, weight_scale=1e-2, reg=1e-4)
        kwargs['learning_rate'] = lr
        Train = trainer(model=model, data=data, kwargs=kwargs.copy())
        model, info = Train.train()
        if best_val_acc < info[0]:
            best_val_acc = info[0]
            best_model = model
            best_kwargs = kwargs.copy()
            best_info = info
    print("best hyparameter: ", best_kwargs)
    get_test_acc(test_data, best_model)
    plot_loss_acc(best_kwargs, best_info)

def temp():
    # config model
    model = logistic_model(input_dims=size_voca, class_nums=num_classes, weight_scale=1e-2, reg=1e-4)

    # Training
    Train = trainer(model=model, data=data, kwargs=kwargs.copy())
    model, info = Train.train()
    
    # plot
    plot_loss_acc(kwargs.copy(), info)

    pred_test_y, _ = model.loss(test_x)
    test_acc = np.sum(np.argmax(pred_test_y, axis=1) == np.argmax(test_y, axis=1)) / len(test_x)
    print("test acc: ")
    print(test_acc, test)



if __name__ == "__main__":
     # load data
    data_set, test_set = get_text_classification_datasets()
    # initialize data processor
    data = {}
    dp = data_processor()
    size_voca = dp.generate_vocabulary(data_set.data)

    # split data
    raw_data, num_classes = split_data(split_point=2000, data_set=data_set)

    # process data
    data["train_x"], data["train_y"] = dp.process_data(
        raw_data["train_x"], raw_data["train_y"], num_classes)
    data["val_x"], data["val_y"] = dp.process_data(
        raw_data["val_x"], raw_data["val_y"], num_classes)
    
    # choose learning rate
    choose_learning_rate(data, size_voca, num_classes)

    # SGD MiniBGD FBGD
    # change_batch_size1(data, size_voca, num_classes)
    # change_batch_size2(data, size_voca, num_classes)
    
    # train and test
    raw_test_x = test_set.data
    raw_test_y = test_set.target

    test_data = {}
    test_data['x'], test_data['y'] = dp.process_data(
        raw_test_x, raw_test_y, num_classes)

    learning_rate = [0.5, 0.2, 0.1, 0.05]
    kwargs = kwargs_global.copy()
    kwargs['lr_decay'] = 0.9
    kwargs["lr_decay_freq"] = 50
    kwargs["batch_size"] = 2000
    kwargs["num_epochs"] = 2000
    kwargs["save_freq"] = 5
    kwargs["check_freq"] = 20
    # training_best_model(data, size_voca, num_classes, test_data, kwargs, learning_rate)

    learning_rate = [0.5, 0.2, 0.1, 0.05]
    kwargs = kwargs_global.copy()
    kwargs['lr_decay'] = 0.9
    kwargs["lr_decay_freq"] = 50
    kwargs["batch_size"] = 50
    kwargs["num_epochs"] = 400
    kwargs["save_freq"] = 20
    kwargs["check_freq"] = 4
    # training_best_model(data, size_voca, num_classes, test_data, kwargs, learning_rate)

    learning_rate = [0.5, 0.2, 0.1, 0.05]
    kwargs = kwargs_global.copy()
    kwargs['lr_decay'] = 0.9
    kwargs["lr_decay_freq"] = 50
    kwargs["batch_size"] = 1
    kwargs["num_epochs"] = 200
    kwargs["save_freq"] = 200
    kwargs["check_freq"] = 2
    # training_best_model(data, size_voca, num_classes, test_data, kwargs, learning_rate)