from sklearn.datasets import fetch_20newsgroups

def get_text_classification_datasets():
    # categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
    dataset_train = fetch_20newsgroups(subset='train')
    dataset_test = fetch_20newsgroups(subset='test')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test