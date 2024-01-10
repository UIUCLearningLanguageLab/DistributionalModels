import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self,
                 corpus,
                 block_size,
                 embedding_size,
                 hidden_layer_info_list,
                 weight_init,
                 criterion,
                 device):

        super(SimpleLSTM, self).__init__()
        self.corpus = corpus
        self.hidden_size = hidden_layer_info_list[0][1]

        if embedding_size == 0:
            embedding_weights = torch.eye(corpus.vocab_size)
            self.embedding_layer = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            self.lstm = nn.LSTM(corpus.vocab_size, self.hidden_size, batch_first=True)
        else:
            self.embedding_layer = nn.Embedding(corpus.vocab_size, embedding_size)
            self.lstm = nn.LSTM(embedding_size, self.hidden_size, batch_first=True)

        # LSTM layer


        # Output layer
        self.linear = nn.Linear(self.hidden_size, corpus.vocab_size)

        self.set_device(device)

    def set_device(self, device=None):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x, hidden=None):

        embedding_out = self.embedding_layer(x)
        # LSTM layer
        lstm_out, hidden = self.lstm(embedding_out, hidden)

        # Only take the output from the final timestep
        # You can modify this part to return the output at each timestep
        lstm_out = lstm_out[:, -1, :]

        # Output layer
        out = self.linear(lstm_out)
        return out, hidden

    def get_weights(self, layer):

        if layer == -1:
            tensor = self.linear.weight
        elif layer == 0:
            tensor = self.embedding_layer.weight
        elif layer == 1:
            tensor = self.lstm.weight_ih_l0.t()
        else:
            raise ValueError("Layer must be 0 or -1 if model has only one layer")

        if str(self.device.type) == 'cpu':
            weight_array = tensor.detach().numpy()
        elif self.device.type == 'cuda':
            weight_array = tensor.detach().cpu().numpy()  # Move tensor to CPU before converting
        elif str(self.device.type) == 'mps':
            weight_array = tensor.detach().to('cpu').numpy()
        else:
            raise ValueError("Unrecognized device", self.device.type)

        return weight_array



