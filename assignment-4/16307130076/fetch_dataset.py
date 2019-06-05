from sklearn.datasets import fetch_20newsgroups
import csv
from nltk.tokenize import sent_tokenize, word_tokenize

class myDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.num_classes = len(set(labels))


def fetch_20newsgroups_dataset():
    dataset_train = fetch_20newsgroups(
        subset='train', data_home='./data/')
    dataset_test = fetch_20newsgroups(
        subset='test', data_home='./data/')
    print("In training dataset:")
    print('Samples:', len(dataset_train.data))
    print('Categories:', len(dataset_train.target_names))
    print("In testing dataset:")
    print('Samples:', len(dataset_test.data))
    print('Categories:', len(dataset_test.target_names))
    d_train = myDataset(dataset_train.data, dataset_train.target)
    d_test = myDataset(dataset_test.data, dataset_test.target)
    return d_train, d_test


def fetch_local_data(data_path):
    texts, labels = [], []
    with open(data_path,encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            label = int(line[0]) - 1
            texts.append(text)
            labels.append(label)
    print('Samples:', len(texts),len(labels))
    print('Categories:',len(set(labels)))
    return myDataset(texts, labels)

def get_max_lengths(dataset):
    word_length_list = []
    sent_length_list = []
    for text in dataset.texts:
        sent_list = sent_tokenize(text)
        sent_length_list.append(len(sent_list))

        for sent in sent_list:
            word_list = word_tokenize(sent)
            word_length_list.append(len(word_list))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

def fetch_dataset(dataset_name):
    if dataset_name == '20newsgroups':
        dataset_train, datset_test = fetch_20newsgroups_dataset()
    elif dataset_name == 'ag_news':
        dataset_train = fetch_local_data('./data/ag_news_csv/train.csv')
        datset_test = fetch_local_data('./data/ag_news_csv/test.csv')
    elif dataset_name == 'amazon_review_full':
        dataset_train = fetch_local_data(
            './data/amazon_review_full_csv/train.csv')
        datset_test = fetch_local_data(
            './data/amazon_review_full_csv/test.csv')
    elif dataset_name == 'amazon_review_polarity':
        dataset_train = fetch_local_data(
            './data/amazon_review_polarity_csv/train.csv')
        datset_test = fetch_local_data(
            './data/amazon_review_polarity_csv/test.csv')
    elif dataset_name == 'dbpedia':
        dataset_train = fetch_local_data(
            './data/dbpedia_csv/train.csv')
        datset_test = fetch_local_data(
            './data/dbpedia_csv/test.csv')
    elif dataset_name == 'sougou_news':
        dataset_train = fetch_local_data(
            './data/sogou_news_csv/train.csv')
        datset_test = fetch_local_data(
            './data/sogou_news_csv/test.csv')
    elif dataset_name == 'yahoo_answers':
        dataset_train = fetch_local_data(
            './data/yahoo_answers_csv/train.csv')
        datset_test = fetch_local_data(
            './data/yahoo_answers_csv/test.csv')
    elif dataset_name == 'yelp_review_full':
        dataset_train = fetch_local_data(
            './data/yelp_review_full_csv/train.csv')
        datset_test = fetch_local_data(
            './data/yelp_review_full_csv/test.csv')
    elif dataset_name == 'yelp_review_polarity':
        dataset_train = fetch_local_data(
            './data/yelp_review_polarity_csv/train.csv')
        datset_test = fetch_local_data(
            './data/yelp_review_polarity_csv/test.csv')
    return dataset_train, datset_test
