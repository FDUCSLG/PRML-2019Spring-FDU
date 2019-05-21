from fastNLP import DataSet


data_path = 'tangshi.csv'
ds = DataSet.read_csv(data_path, headers=('正文'))

print(ds[0])
