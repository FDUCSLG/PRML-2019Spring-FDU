import torch, string, re, model, os
from fastNLP import Tester
from fastNLP import Trainer
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
from fastNLP.core.optimizer import Adam
from fastNLP.core.callback import EarlyStopCallback

from sklearn.datasets import fetch_20newsgroups

lr = 0.001
epochs = 50
early_epoch = 10
ratio = 0.1
device = "cuda:0"
batch_size = 8
num_classes = 20
bidirect = True
embed_dim = 128
hidden_dim = 128
num_layers = 2
model_save_path = "./res_model/"
os.makedirs(model_save_path, exist_ok=True)

def main(net, optimizer, train_set, test_set, loss, metric):
	train_data, dev_data = train_set.split(ratio)
	trainer = Trainer(model=net, loss=loss, optimizer=optimizer, n_epochs=epochs,
					  train_data=train_data, dev_data=dev_data, metrics=metric,
					  device=device, save_path=model_save_path, print_every=200,
					  use_tqdm=False, callbacks=[EarlyStopCallback(early_epoch)])
	trainer.train()
	
	tester = Tester(model=net, data=test_set, metrics=metric, device=device)
	tester.test()



def get_data(dataset):
	n = len(dataset.data)
	data_set = DataSet()
	for i in range(n):
		data_set.append(Instance(raw_sentence=dataset.data[i], target=int(dataset.target[i])))

	data_set.apply(lambda x: x['raw_sentence'].lower(), new_field_name = 'sentence')
	data_set.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x['sentence']), new_field_name = 'sentence')
	data_set.apply(lambda x: re.sub('[%s]' % re.escape(string.whitespace), ' ', x['sentence']), new_field_name = 'sentence')
	data_set.apply(lambda x: x['sentence'].split(), new_field_name = 'words')

	return data_set


if __name__ == "__main__":
	train_dataset = fetch_20newsgroups(subset='train', data_home='./')
	test_dataset = fetch_20newsgroups(subset='test', data_home='./')
	train_set = get_data(train_dataset)
	test_set = get_data(test_dataset)

	vocab = Vocabulary(min_freq=10).from_dataset(train_set, field_name='words')
	vocab.index_dataset(train_set, field_name='words',new_field_name='input')
	vocab.index_dataset(test_set, field_name='words',new_field_name='input')
	vocab_size = len(vocab)
	train_set.set_input('input')
	train_set.set_target('target')
	test_set.set_input('input')
	test_set.set_target('target')

	model_name = "RNN"
	if model_name == "CNN":
		net = model.CNN(vocab_size, embed_dim, num_classes)
	else:
		net = model.RNN(vocab_size, embed_dim, num_classes, hidden_dim,
						num_layers, bidirect)

	print(bidirect, num_layers)

	optimizer = Adam(lr=lr, weight_decay=1e-4)
	loss = CrossEntropyLoss(pred="output", target="target")
	acc = AccuracyMetric(pred="output", target="target")
	main(net, optimizer, train_set, test_set, loss, acc)




