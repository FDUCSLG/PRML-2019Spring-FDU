
dataset_path = "./dataset/"
train_data_path = dataset_path + "train_data.pkl"
dev_data_path = dataset_path + "validate_data.pkl"
test_data_path =  dataset_path + "test_data.pkl"
vocab_path = dataset_path + "vocab.pkl"
save_path = "./checkpoint"
model = "RNN"
rnn_type = "max" #max、min、mean、attention

class_num = 20
learning_rate = 1e-3
epoch = 200
batch_size = 16
intput_size = 128
hidden_size = 256
validate_every = 50
patience = 10
