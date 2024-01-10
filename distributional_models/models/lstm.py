import torch
import torch.nn as nn
from .neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
    def __init__(self,
                 corpus,
                 embedding_size,
                 hidden_size,
                 weight_init,
                 device):

        super(LSTM, self).__init__(corpus, device)
        self.model_type = 'lstm'
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        

        self.define_network()
        self.set_device(device)

    def define_network(self):

        if self.embedding_size == 0:
            embedding_weights = torch.eye(self.corpus.vocab_size)
            self.layer_dict['embedding'] = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            self.layer_dict['lstm'] = nn.LSTM(self.corpus.vocab_size, self.hidden_size, batch_first=True)
        else:
            self.layer_dict['embedding'] = nn.Embedding(self.corpus.vocab_size, self.embedding_size)
            self.layer_dict['lstm'] = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.corpus.vocab_size)

    def forward(self, x):

        embedding_out = self.layer_dict['embedding'](x)
        # LSTM layer
        lstm_out, self.hidden_dict['lstm'] = self.layer_dict['lstm'](embedding_out, self.hidden_dict['lstm'])

        # Only take the output from the final timestep
        # You can modify this part to return the output at each timestep
        lstm_out = lstm_out[:, -1, :]

        # Output layer
        out = self.layer_dict['output'](lstm_out)

        return out
