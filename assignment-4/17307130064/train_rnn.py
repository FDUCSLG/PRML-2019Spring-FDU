from create_dataset import create_dataset

from rnn import lstm
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric


vocab, train_data, dev_data, test_data = create_dataset()

model = lstm(vocab_size=len(vocab), embedding_length=50, hidden_size=32, output_size=5)
model.cuda()

loss = CrossEntropyLoss(pred='pred', target='target')
metrics = AccuracyMetric(pred='pred', target='target')

trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics, save_path='./', device=0, n_epochs=20)
trainer.train()