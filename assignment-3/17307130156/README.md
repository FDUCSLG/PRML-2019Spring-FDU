# Chinese Poetry Generator

```
.
├── model.py, train.py, run.py: RNN Model based on official LSTM
├── custom_lstm.py, custom_model.py, custom_train.py, custom_run.py: RNN Model based on custom LSTM
|
├── utils.py: Data handler
├── vocabulary.py: Vocabulary handler
|
├── checkpoints: Model checkpoints
├── raw_data: Chinese poetry jsons
|
├── docs: Report in detail
└── README.md
```

Json data and checkpoints are not included, thus cannot reproducing my result. But if you want to reproduce my work, please check my github repos.

Dependency: Pytorch

## Usage

I have implemented a custom LSTM, which is almost the same as official one.

### Official LSTM

Train
```bash
python3 train.py
```

The training will automatically stop when the perplexity doesn't decrease any more (after about 20 epochs).

Run

```bash
python3 run.py
```

### Custom LSTM

Train

```bash
python3 custom_train.py
```

Run

```bash
python3 custom_run.py
```

