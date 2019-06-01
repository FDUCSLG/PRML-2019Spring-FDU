import torch
from torch import nn
import torch.nn.functional as F
import fastNLP.modules.encoder as encoder
from fastNLP.modules import utils
from fastNLP.modules.encoder.bert import BertModel
from fastNLP.models.base_model import BaseModel

class MyBertModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dropout_prob, num_labels, 
                 use_word2vec=False, embedding_weight_path=None):
        super(MyBertModel, self).__init__()
        self.num_labels = num_labels
        
        self.bert = BertModel(vocab_size=input_size, hidden_size=hidden_size,num_hidden_layers=2,num_attention_heads=2,
                              intermediate_size=512, hidden_dropout_prob=hidden_dropout_prob,attention_probs_dropout_prob=0.1,
                              max_position_embeddings=2500, 
                             )
        if use_word2vec:
            self.bert.embeddings.word_embeddings.weight.data.copy_(
                torch.from_numpy(
                    load_embedding_matrix(embedding_weight_path)
                ))

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_data, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return {"output": logits}
    """ 
    def predict(self, input_data, token_type_ids=None, attention_mask=None):
        logits = self.forward(input_data, token_type_ids, attention_mask)
        return {"output": torch.argmax(logits, dim=-1)}
    """

def load_embedding_matrix(embedding_weight_path):
    import _pickle as pickle
    return pickle.load(open(embedding_weight_path, 'rb'))
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_dim, 
                 use_word2vec=False, embedding_weight_path=None):
        super(LSTMModel, self).__init__()
        self.hidden_dim = 256
        self.embeddings = nn.Embedding(input_size, embedding_dim)
        if use_word2vec:
            self.embeddings.weight.data.copy_(
                torch.from_numpy(
                    load_embedding_matrix(embedding_weight_path)
                ))

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_data, hidden=None):
        batch_size, seq_len = input_data.size()
        if hidden is None:
            h_0 = input_data.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_data.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden 
        # size: (batch_size, seq_len, embeding_dim)
        embeds = self.embeddings(input_data)
        # output size: (batch_size, seq_len, hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # max_pool
        #output = -output 
        output_pool = F.max_pool1d(output.permute(0,2,1), output.size(1)).squeeze(2)
        #output = -output
        
        # mean_pool
        #output_pool = torch.sum(output, dim=1) / seq_len

        output = self.linear1(output_pool)
        """ 
        output = self.fc1(output_pool)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        """
        return {'output': output}


class B_LSTMModel(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_dim,
                 use_word2vec=False, embedding_weight_path=None):
        super(B_LSTMModel, self).__init__()
        self.hidden_dim = 256
        self.embeddings = nn.Embedding(input_size, embedding_dim)
        if use_word2vec:
            self.embeddings.weight.data.copy_(
                torch.from_numpy(
                    load_embedding_matrix(embedding_weight_path)
                ))

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_dim*2, vocab_size)

    def forward(self, input_data, hidden=None):
        batch_size, seq_len = input_data.size()
        if hidden is None:
            h_0 = input_data.data.new(2*2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_data.data.new(2*2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden 
        # size: (batch_size, seq_len, embeding_dim)
        embeds = self.embeddings(input_data)
        # output size: (batch_size, seq_len, hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # max pool
        #output = -output 
        #output_pool = F.max_pool1d(output.permute(0,2,1), output.size(1)).squeeze(2) 
        #output_pool = -output_pool
        # mean pooling
        #output_pool = torch.sum(output, dim=1) / seq_len
        output_pool = output[:,-1,:]

        output = self.linear1(output_pool)
        return {'output': output}



class CNNModel(torch.nn.Module):

    def __init__(self, input_size,
                 vocab_size,
                 embedding_dim,
                 use_word2vec = False,
                 embedding_weight_path = None,
                 kernel_nums=(100, 100, 100),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNModel, self).__init__()

        # no support for pre-trained embedding currently
        self.embeddings = torch.nn.Embedding(input_size, embedding_dim)
        if use_word2vec:
            self.embeddings.weight.data.copy_(
                torch.from_numpy(
                    load_embedding_matrix(embedding_weight_path) 
                ))

        self.conv_pool = encoder.ConvMaxpool(
            in_channels=embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(sum(kernel_nums), sum(kernel_nums))
        self.fc2 = torch.nn.Linear(sum(kernel_nums), vocab_size)
        self.fc = torch.nn.Linear(sum(kernel_nums), vocab_size)
    def forward(self, input_data):
        x = self.embeddings(input_data)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)
        """
        x = self.fc1(x)  # [N,C] -> [N, N_class]
        x = F.relu(x)
        x = self.fc2(x)
        """
        return {'output': x}

