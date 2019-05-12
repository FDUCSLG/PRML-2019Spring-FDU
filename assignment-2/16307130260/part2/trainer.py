import numpy as np
class trainer:
    def __init__(self, model, data, kwargs):
        self.model = model
        self.best_params = self.model.params.copy()

        self.train_x = data["train_x"]
        self.train_y = data["train_y"]
        self.has_val = ("val_x" in data.keys())
        if self.has_val:
            self.val_x = data["val_x"]
            self.val_y = data["val_y"]

        self.learning_rate = kwargs.pop('learning_rate', 1e-2)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop("batch_size", 1)


        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.delta_loss = kwargs.pop('delta_loss', 0)
        self.beta = kwargs.pop('beta', 0)
        # when use delta loss as the terminating condition
        if self.delta_loss != 0:
            self.num_epochs = 23333333

        self.lr_decay_freq = kwargs.pop('lr_decay_freq', 100)
        # save loss per save_freq steps
        self.save_freq = kwargs.pop('save_freq', 1)
        # check acc per check_freq epoches
        self.check_freq = kwargs.pop('check_freq', 1)


    # sgd step
    def _step(self, x, y):
        loss, grads = self.model.loss(x, y)
        for key in grads.keys():
            self.model.params[key] -= self.learning_rate * grads[key]
        return loss
    
    # check the model acc in training set and validation set
    def _check_acc(self):
        train_acc = 0
        val_acc = 0

        scores_train, _ = self.model.loss(self.train_x)
        pred_train_y = np.argmax(scores_train, axis=1)
        train_acc = np.sum(pred_train_y == np.argmax(self.train_y, axis=1)) / pred_train_y.shape[0]

        if self.has_val == False:
            return train_acc, val_acc
        
        scores_val, _ = self.model.loss(self.val_x)
        pred_val_y = np.argmax(scores_val, axis=1)
        val_acc = np.sum(pred_val_y == np.argmax(self.val_y, axis=1)) / pred_val_y.shape[0]
        return train_acc, val_acc

    def train(self):
        best_val_acc = 0
        loss_history = []
        train_acc_his = []
        val_acc_his = []

        cnt_steps = 0
        

        # set previous as infinite 
        loss_pre = 10
        dloss = 0

       
        rng = np.random.RandomState(2333)
        epoch = 0 
        while epoch < self.num_epochs:
            # check the acc and save history
            if epoch % self.check_freq == 0:
                train_acc, val_acc = self._check_acc()
                print("epoch: " + str(epoch))
                
                print("train_acc: " + str(train_acc))
                train_acc_his.append(train_acc)

                if self.has_val:
                    print("val_acc: " + str(val_acc))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        self.best_params = self.model.params.copy()
                    val_acc_his.append(val_acc)

            # random shuffle training set in every epoch
            rand_ind = np.arange(self.train_x.shape[0])
            rng.shuffle(rand_ind)
            self.train_x = self.train_x[rand_ind]
            self.train_y = self.train_y[rand_ind]

            
            for i in range(0, self.train_x.shape[0], self.batch_size):
                # get batch training set 
                x_train_batch = self.train_x[i:i + self.batch_size]
                y_train_batch = self.train_y[i:i + self.batch_size]
                loss = self._step(x_train_batch, y_train_batch)

                # output loss info
                cnt_steps += 1
                if cnt_steps % self.save_freq == 0:
                    print("steps: " + str(cnt_steps) + ", loss: " + str(loss))
                    loss_history.append(loss)
                
                # terminate when change of loss is small
                dloss = self.beta * dloss + (1 - self.beta) * abs(loss_pre - loss)
                if self.delta_loss != 0 and dloss < self.delta_loss:
                    epoch = self.num_epochs + 1
                    break
                loss_pre = loss

            # decay lr
            if epoch % self.lr_decay_freq == 0:
                self.learning_rate *= self.lr_decay
            epoch += 1
        
        if self.has_val:
            self.model.params = self.best_params.copy()
        info_list = [best_val_acc, loss_history, train_acc_his, val_acc_his]
        return self.model, info_list