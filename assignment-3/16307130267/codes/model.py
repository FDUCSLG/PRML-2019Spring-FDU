import numpy as np
from model_parts import Embedding, LSTM, TemporalFC, FC

np.random.seed(233)

class GeneratingLSTM(object):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, dtype=np.float32):
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.params = {}

        vocab_size = len(word_to_idx)

        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        dim_mul = 4
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        self.embedding = Embedding()
        self.lstm = LSTM()
        self.fc = TemporalFC()

    def forward(self, sentence_in, hidden=None):
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        if hidden is None:
            hidden0 = np.random.randn(sentence_in.shape[0], self.hidden_dim)
        else:
            hidden0 = hidden
        
        word_embedding_out = self.embedding.forward(sentence_in, W_embed) # [bs, 125-1, w=128]
        lstm_out, lstm_hidden = self.lstm.forward(word_embedding_out, hidden0, Wx, Wh, b) # [bs, 125-1, h=128]
        temporal_affine_out = self.fc.forward(lstm_out, W_vocab, b_vocab)
        return temporal_affine_out, lstm_hidden

    def backward(self, dtemporal_affine_out):
        grads = {}
        dlstm_out, grads['W_vocab'], grads['b_vocab'] = self.fc.backward(dtemporal_affine_out)
        dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = self.lstm.backward(dlstm_out)
        grads['W_embed'] = self.embedding.backward(dword_embedding_out)
        return grads

    def clear_cache(self):
        self.embedding.cache = None
        self.lstm.cache = None
        self.fc.cache = None
