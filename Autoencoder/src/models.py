import torch
import numpy as np
from torch import nn, optim

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value):
        
        attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        attention = nn.functional.softmax(attention, dim=1)

        out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return out, attention

class Encoder_attention(nn.Module):
    def __init__(self, n_features=1, hidden_dim=128, value_size=128,key_size=128):
        super(Encoder_attention, self).__init__()

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_dim,
            num_layers = 2,
            bidirectional = True,
            batch_first = True
        )
        
        self.key_network = nn.Linear(hidden_dim*2, key_size)
        self.value_network = nn.Linear(hidden_dim*2, value_size)

    def forward(self, x):
        x = self.lstm(x)[0]
        
        keys = self.key_network(x)
        value = self.value_network(x)
        
        return keys, value

class Decoder_attention(nn.Module):

    def __init__(self, n_features, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Decoder_attention, self).__init__()

        self.lstm1 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()
        
        if (isAttended == True):
            self.output_layer = nn.Linear(key_size+value_size, n_features)
        else:
            self.output_layer = nn.Linear(key_size, n_features)

    def forward(self, keys, values):
        length = values.shape[1]
        predictions = []
        hidden_states = [None, None]
        
        context = values[:, 0, :]
        
        for i in range(length):
            inp = context
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])
            
            output = hidden_states[1][0]
            if (self.isAttended == True):
                context, attention = self.attention(output, keys, values)
                prediction = self.output_layer(torch.cat([output, context], dim=1))
            else:
                prediction = self.output_layer(output)
                
            predictions.append(prediction.unsqueeze(1))
            
        return torch.cat(predictions, dim=1)
    
class Encoder_lstm(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder_lstm, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder_lstm(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder_lstm, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder_lstm(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder_lstm(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
class Autoencoder_attention(nn.Module):

    def __init__(self, n_features, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Autoencoder_attention, self).__init__()

        self.encoder = Encoder_attention(n_features, hidden_dim, value_size, key_size)
        self.decoder = Decoder_attention(n_features, hidden_dim, value_size, key_size, isAttended)

    def forward(self, x):
        keys, values = self.encoder(x)
        out = self.decoder(keys, values)

        return out
