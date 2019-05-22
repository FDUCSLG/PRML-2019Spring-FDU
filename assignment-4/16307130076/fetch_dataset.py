from sklearn.datasets import fetch_20newsgroups

def fetch_20newsgroups_dataset():
    dataset_train = fetch_20newsgroups(
        subset='train',data_home='../../..')
    dataset_test = fetch_20newsgroups(
        subset='test',data_home='../../..')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    return dataset_train, dataset_test

def fetch_dataset(dataset_name):
    if dataset_name == '20newsgroups':
        dataset_train, datset_test = fetch_20newsgroups_dataset()
    return dataset_train,datset_test