
dataset_path = "./dataset/"
data_path = dataset_path + "poetry.csv"
train_data_path = dataset_path + "train_data.pkl"
dev_data_path = dataset_path + "validate_data.pkl"
vocab_path = dataset_path + "vocab.pkl"
save_path = "./checkpoint"
learning_rate = 1e-3
epoch = 200
batch_size = 128
MAXLEN = 120
MAX_GEN_SIZE = 8
intput_size = 512
hidden_size = 512
validate_every = 50
patience = 10
temperature = 1