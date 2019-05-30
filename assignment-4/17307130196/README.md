# README

This is an implementation of CNN and RNN to finish the task in Assignment 2. 

### Structure of The FileFolder

```
Assignment 4
+--model
|  +--CNNclassifier.py
|  +--RNNclassifier,py
+--data
|  +--assis.py
|  +--preprocess.py
|  +--data.csv
|  +--dataset.py
+--report.pdf
+--README.py (This file)
+--run.py
```

**Some remarks**

- `model`: LSTM model
- `run.py`: Key file, the trainer
- `data`: Some preprocessing work and datasets.

### How to use?

```
python run.py
```

And you will receive guiding words to help you for further explorations. You can modifying the dataset you'd like to use in `data/preprocess.py` . All trainers lie in `run.py`. You could go to the `model/` to get details of the implementation of the CNN and RNN. fastNLP version is almost the same as the model in the tutorial doc, so it will not be listed here.

**Enjoy yourself!**