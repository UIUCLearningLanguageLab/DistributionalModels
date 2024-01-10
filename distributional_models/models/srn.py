import torch
import torch.nn as nn
from .neural_network import NeuralNetwork


class SRN(NeuralNetwork):
    def __init__(self,
                 corpus,
                 embedding_size,
                 hidden_size,
                 weight_init,
                 device):

        super(SRN, self).__init__(corpus, device)
        self.model_type = 'srn'
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.define_network()
        self.set_device(device)

    def define_network(self):

        if self.embedding_size == 0:
            embedding_weights = torch.eye(self.corpus.vocab_size)
            self.layer_dict['embedding'] = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            self.layer_dict['srn'] = nn.RNN(self.corpus.vocab_size, self.hidden_size, batch_first=True)
        else:
            self.layer_dict['embedding'] = nn.Embedding(self.corpus.vocab_size, self.embedding_size)
            self.layer_dict['srn'] = nn.RNN(self.embedding_size, self.hidden_size, batch_first=True)

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.corpus.vocab_size)

    def forward(self, x, hidden=None):

        embedding_out = self.layer_dict['embedding'](x)
        # LSTM layer
        srn_out, self.hidden_dict['srn'] = self.layer_dict['srn'](embedding_out, self.hidden_dict['srn'])

        # Only take the output from the final timestep
        # You can modify this part to return the output at each timestep
        srn_out = srn_out[:, -1, :]

        # Output layer
        out = self.layer_dict['output'](srn_out)
        return out
