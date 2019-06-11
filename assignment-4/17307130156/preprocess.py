from sklearn.datasets import fetch_20newsgroups

from fastNLP import DataSet, Instance, Vocabulary, Const
from fastNLP import CrossEntropyLoss, AccuracyMetric, Trainer, Tester
from fastNLP.models import CNNText

# Convert to fastNLP DataSet
def to_dataset(datas, targets):

    instances = []
    for data, target in zip(datas, targets):
        data = data.lower()
        words = data.split()
        instances.append(Instance(raw_sentence=data, words=words, target=int(target)))

    dataset = DataSet(instances)
    return dataset


# Get trainset, devset, testset and vacabulary
def get_train_dev_test_vocab():

    dataset_train = fetch_20newsgroups(subset='train', data_home='../../../')
    dataset_test = fetch_20newsgroups(subset='test', data_home='../../../')
    # dataset_train, dataset_test = get_text_classification_datasets()
    
    train_data = dataset_train.data
    train_target = dataset_train.target
    test_data = dataset_test.data
    test_target = dataset_test.target
    print (f'train dataset: {len(train_data)}')
    print (f'test dataset: {len(test_data)}')
    
    train_dataset = to_dataset(train_data, train_target)
    test_dataset = to_dataset(test_data, test_target)

    vocab = Vocabulary(min_freq=10).from_dataset(train_dataset, field_name='words')
    print (f'Vocab size: {len(vocab)}')

    vocab.index_dataset(train_dataset, field_name='words', new_field_name='words')
    vocab.index_dataset(test_dataset, field_name='words', new_field_name='words')

    train_dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    test_dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')


    # Rename to suit inbuilt Model in fastNLP
    train_dataset.rename_field('words', Const.INPUT)
    train_dataset.rename_field('seq_len', Const.INPUT_LEN)
    train_dataset.rename_field('target', Const.TARGET)
    train_dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    train_dataset.set_target(Const.TARGET)

    test_dataset.rename_field('words', Const.INPUT)
    test_dataset.rename_field('seq_len', Const.INPUT_LEN)
    test_dataset.rename_field('target', Const.TARGET)
    test_dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    test_dataset.set_target(Const.TARGET)

    # Split into development dataset
    train_dataset, dev_dataset = train_dataset.split(0.1)

    return train_dataset, dev_dataset, test_dataset, vocab


if __name__ == '__main__':

    # Usage
    train_data, dev_data, test_data, vocab = get_train_dev_test_vocab()

    model_cnn = CNNText((len(vocab),50), 
                        num_classes=20, 
                        padding=2, 
                        dropout=0.1)

    loss = CrossEntropyLoss(pred=Const.OUTPUT, 
                            target=Const.TARGET)

    metrics = AccuracyMetric(pred=Const.OUTPUT, 
                             target=Const.TARGET)

    trainer = Trainer(model=model_cnn,
                      train_data=train_data, 
                      dev_data=dev_data, 
                      loss=loss, 
                      metrics=metrics)

    trainer.train()

    tester = Tester(test_data, 
                    model_cnn, 
                    metrics=AccuracyMetric())

    tester.test()

    



