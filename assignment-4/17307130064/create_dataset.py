from sklearn.datasets import fetch_20newsgroups
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const

def create_dataset():
        categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'rec.motorcycles']
        # categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'rec.motorcycles', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale']
        # categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')

        dataset = DataSet()

        for i in range(len(newsgroups_train.data)):
            if len(newsgroups_train.data[i]) <= 3000:
                dataset.append(Instance(raw_sentence=newsgroups_train.data[i], target=int(newsgroups_train.target[i])))
        for i in range(len(newsgroups_test.data)):
            if len(newsgroups_test.data[i]) <= 3000:
                dataset.append(Instance(raw_sentence=newsgroups_test.data[i], target=int(newsgroups_test.target[i])))

        dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
        dataset.apply(lambda x: x['sentence'].split(), new_field_name='words')
        dataset.apply(lambda x: len(x['words']), new_field_name='seq_len')

        vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
        vocab.index_dataset(dataset, field_name='words', new_field_name='words')

        dataset.set_input('words', 'seq_len')
        dataset.set_target('target')

        train_dev_data, test_data = dataset.split(0.1)
        train_data, dev_data = train_dev_data.split(0.1)

        return vocab, train_data, dev_data, test_data