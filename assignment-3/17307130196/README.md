# README

This is an implementation of LSTM to write tang poems automatically. We'd call it **PeomProducer**. The technique details and some mathematical proof will be left to the report. 

### Structure of The FileFolder

```
Writing-A-Poem
+--model
|  +--LSTM.py
+--data
|  +--embedding.word
|  +--langconv.py
|  +--preprocess.py
|  +--test.json
|  +--zh_wiki.py
|  +--poems
|     +--poet.tang.xxxx.json
+--figs
|  +--bptt.png
|  +--optim.png
|  +--pretrain.png
|  +--trick.png
+--report.pdf
+--README.py (This file)
+--test.py
+--run.py
```

**Some remarks**

- `model`: LSTM model
- `figs`: The figures drawn in the experiment
- `run.py`: Key file, the trainer

- `data`: Some preprocessing work and datasets.

### How to use?

```
python run.py
```

And you will receive guiding words to help you for further explorations. You can modifying the dataset you'd like to use in `data/preprocess.py` . All trainers lie in `run.py`. You could go to the `model/LSTM.py` to get details of the implementation of the LSTM. There's also a numpy version, but it's a quite basic implement since it is only used to show how the BPTT of LSTM. 

**Enjoy yourself!**

