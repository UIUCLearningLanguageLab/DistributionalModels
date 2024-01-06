import torch
import numpy as np
import time
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ..corpora.corpus import Corpus
import torch.nn.utils.rnn as rnn_utils


class NeuralNetwork(torch.nn.Module):
    def __init__(self, corpus, embedding_size, hidden_layer_info_list, weight_init, criterion, device=None,
                 mask_unknown_loss=True):
        super(NeuralNetwork, self).__init__()
        self.layer_list = None
        self.embedding_size = embedding_size
        self.corpus = corpus
        self.vocab_index_dict = corpus.vocab_index_dict
        self.vocab_size = corpus.vocab_size
        self.weight_init = weight_init
        self.hidden_layer_info_list = hidden_layer_info_list
        self.layer_type_list = None
        self.layer_size_list = None
        self.layer_state_list = None
        self.criterion = None
        self.mask_unknown_loss = mask_unknown_loss
        self.optimizer = None
        self.device = None
        self.is_recurrent = False
        self.has_transformer = False
        self.unknown_token = corpus.unknown_token

        self.init_model()
        self.set_device(device)
        self.set_criterion(criterion)
        self.define_layers()
        self.to(self.device)

    def init_model(self):
        # TODO only do this in the case where the model needs an unknown and one doesnt exist
        if self.unknown_token is None:
            self.unknown_token = "<unk>"
            if "<unk" not in self.corpus.vocab_index_dict:
                self.corpus.add_token_to_vocab(self.unknown_token)

    def set_device(self, device=None):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    # TODO add the criterion for multi label comparison
    def set_criterion(self, criterion):
        if criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid criterion")

    def set_optimizer(self, optimizer, learning_rate):
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def define_layers(self):
        self.layer_type_list = []
        self.layer_list = []
        self.layer_size_list = []

        # this adds an embedding layer if there is one
        if self.embedding_size != 0:
            self.layer_list.append(torch.nn.Embedding(self.vocab_size, self.embedding_size))
            self.layer_size_list.append(self.embedding_size)
            previous_layer_size = self.embedding_size
            self.layer_type_list.append("embedding")
        else:
            previous_layer_size = self.vocab_size

        if len(self.hidden_layer_info_list) > 0:
            for i in range(len(self.hidden_layer_info_list)):
                if self.hidden_layer_info_list[i][0] == "linear":
                    layer_list, type_list, size_list, previous_layer_size = self.create_linear_layer(
                        previous_layer_size, self.hidden_layer_info_list[i][1])
                elif self.hidden_layer_info_list[i][0] == "lstm":
                    layer_list, type_list, size_list, previous_layer_size = self.create_lstm_layer(
                        previous_layer_size, self.hidden_layer_info_list[i][1])
                    self.is_recurrent = True
                elif self.hidden_layer_info_list[i][0] == "srn":
                    layer_list, type_list, size_list, previous_layer_size = self.create_srn_layer(
                        previous_layer_size, self.hidden_layer_info_list[i][1])
                    self.is_recurrent = True
                elif self.hidden_layer_info_list[i][0] == 'gpt':
                    layer_list, type_list, size_list, previous_layer_size = self.create_gpt_layer(previous_layer_size,
                                                                                                  self.hidden_layer_info_list[i][1])
                    self.has_transformer = True
                else:
                    raise ValueError(f"Unrecognized hidden layer type {self.hidden_layer_info_list[i][0]}")

                self.layer_type_list += type_list
                self.layer_size_list += size_list
                self.layer_list += layer_list

        # adds the output layer
        self.layer_list.append(torch.nn.Linear(previous_layer_size, self.vocab_size))
        self.layer_size_list.append(self.vocab_size)
        self.layer_type_list.append("linear")

        self.layer_list = torch.nn.ModuleList(self.layer_list)

    @staticmethod
    def create_linear_layer(previous_layer_size, layer_size, activation_function="relu"):
        size_list = [layer_size]
        layer_list = [torch.nn.Linear(previous_layer_size, layer_size)]
        type_list = ["linear"]
        if activation_function is not None:
            if activation_function == 'relu':
                type_list.append('relu')
                layer_list.append(torch.nn.ReLU())
                size_list.append(layer_size)
            else:
                raise ValueError(f"ERROR: Unrecognized activation function {activation_function}")
        previous_size = layer_size
        return layer_list, type_list, size_list, previous_size

    @staticmethod
    def create_lstm_layer(previous_layer_size, layer_size):
        layer_list = [torch.nn.LSTM(input_size=previous_layer_size, hidden_size=layer_size, batch_first=True)]
        previous_size = layer_size
        size_list = [layer_size]
        type_list = ["lstm"]
        return layer_list, type_list, size_list, previous_size

    @staticmethod
    def create_srn_layer(previous_layer_size, layer_size):
        layer_list = [torch.nn.RNN(input_size=previous_layer_size, hidden_size=layer_size, batch_first=True)]
        previous_size = layer_size
        size_list = [layer_size]
        type_list = ["srn"]
        return layer_list, type_list, size_list, previous_size

    def create_gpt_layer(self, previous_layer_size, layer_size, block_size):
        layer_list = []
        type_list = []
        size_list = []
        return layer_list, type_list, size_list, previous_layer_size

    #     num_heads = layer_size[0]  # 8
    #     head_size = layer_size[1]  # 4
    #     total_heads = num_heads * head_size  # 32
    #
    #     heads = []
    #     for i in range(num_heads):
    #         key = torch.nn.Linear(previous_layer_size, head_size, bias=False)
    #         query = torch.nn.Linear(previous_layer_size, head_size, bias=False)
    #         value = torch.nn.Linear(previous_layer_size, head_size, bias=False)
    #
    #
    #     self.heads = torch.nn.ModuleList([Head(head_size, embed_size, block_size) for i in range(num_heads)])
    #
    #
    #     self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    #     pre_mask_attention_weights = None
    #

    #
    # class Head(nn.Module):
    #     def __init__(self, head_size, embed_size, block_size):
    #         super().__init__()
    #         self.key = nn.Linear(embed_size, head_size, bias=False)
    #         self.query = nn.Linear(embed_size, head_size, bias=False)
    #         self.value = nn.Linear(embed_size, head_size, bias=False)
    #         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    #         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #         self.pre_mask_attention_weights = None
    #
    #     def forward(self, x):
    #         B, T, C = x.shape
    #         k = self.key(x)  # (B ,T, C)
    #         q = self.query(x)  # (B ,T, C)
    #         v = self.value(x)  # (B ,T, C)
    #         # wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention, original in paper "attention is all you need"
    #         wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
    #         self.pre_mask_attention_weights = wei.clone()
    #         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
    #         wei = F.softmax(wei, dim=1)  # (B, T, T)
    #
    #         out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
    #         return out
    #
    # def forward(self, x):
    #     attention_weights_list = []
    #     head_outputs = []
    #     for head in self.heads:
    #         B, T, C = x.shape
    #         k = self.key(x)  # (B ,T, C)
    #         q = self.query(x)  # (B ,T, C)
    #         v = self.value(x)  # (B ,T, C)
    #         # wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention, original in paper "attention is all you need"
    #         wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
    #         self.pre_mask_attention_weights = wei.clone()
    #         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
    #         wei = F.softmax(wei, dim=1)  # (B, T, T)
    #
    #         out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
    #         head_outputs.append(head_output)
    #
    #         # Assuming pre_mask_attention_weights are stored in each head
    #         attention_weights_list.append(head.pre_mask_attention_weights)
    #
    #     # Concatenating the outputs from all heads
    #     combined_output = torch.cat(head_outputs, dim=-1)
    #
    #     # Average the attention weights across all heads
    #     combined_attention_weights = torch.cat(attention_weights_list, dim=-1)
    #     return combined_output, combined_attention_weights
    #
    # class MultiHeadAttention(nn.Module):
    #     def __init__(self, num_heads, head_size, embed_size, block_size):
    #         super().__init__()
    #         self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for i in range(num_heads)])
    #
    #     def forward(self, x):
    #         attention_weights_list = []
    #         head_outputs = []
    #         for head in self.heads:
    #             head_output = head(x)
    #             head_outputs.append(head_output)
    #
    #             # Assuming pre_mask_attention_weights are stored in each head
    #             attention_weights_list.append(head.pre_mask_attention_weights)
    #
    #         # Concatenating the outputs from all heads
    #         combined_output = torch.cat(head_outputs, dim=-1)
    #
    #         # Average the attention weights across all heads
    #         combined_attention_weights = torch.cat(attention_weights_list, dim=-1)
    #         return combined_output, combined_attention_weights

    def init_weights(self):
        for i in range(len(self.layer_list)):
            self.layer_list[i].weight.data.uniform_(-self.embed_init_range, self.embed_init_range)

    def init_layer_states(self, batch_size=1):
        self.layer_state_list = []
        for i in range(len(self.layer_list)):
            if self.layer_type_list[i] == "embedding":
                layer_state = torch.zeros(self.layer_size_list[i]).to(self.device)
            elif self.layer_type_list[i] == "linear":
                layer_state = torch.zeros(self.layer_size_list[i]).to(self.device)
            elif self.layer_type_list[i] == "lstm":
                hidden_state = torch.zeros(1, batch_size, self.layer_size_list[i]).to(self.device)
                cell_state = torch.zeros(1, batch_size, self.layer_size_list[i]).to(self.device)
                layer_state = (hidden_state, cell_state)
            elif self.layer_type_list[i] == "srn":
                layer_state = torch.zeros(1, batch_size, self.layer_size_list[i]).to(self.device)
            elif self.layer_type_list[i] == "relu":
                layer_state = torch.zeros(self.layer_size_list[i]).to(self.device)
            else:
                raise ValueError(f"Unrecognized hidden layer type {self.layer_type_list[i]}")

            self.layer_state_list.append(layer_state)

    def resize_states(self, states, batch_size):
        """Resize LSTM hidden and cell states to the new batch size."""
        resized_states = []

        for i in range(len(self.layer_list)):
            state = states[i]

            if isinstance(self.layer_list[i], torch.nn.LSTM):
                # Resize each tensor in the tuple
                resized_state = []
                for s in state:
                    # State shape for LSTM: (num_layers, batch_size, hidden_size)
                    if s.size(1) != batch_size:
                        # Create a new state tensor with the correct batch size
                        new_state = torch.zeros(s.size(0), batch_size, s.size(2), device=self.device)
                        # Copy the values from the old state tensor
                        min_batch_size = min(s.size(1), batch_size)
                        new_state[:, :min_batch_size, :] = s[:, :min_batch_size, :]
                        resized_state.append(new_state)
                    else:
                        resized_state.append(s)
                resized_states.append(tuple(resized_state))
            elif isinstance(self.layer_list[i], torch.nn.RNN):
                s = state
                if s.size(1) != batch_size:
                    new_state = torch.zeros(s.size(0), batch_size, s.size(2), device=self.device)
                    min_batch_size = min(s.size(1), batch_size)
                    new_state[:, :min_batch_size, :] = s[:, :min_batch_size, :]
                    resized_states.append(new_state)
                else:
                    resized_states.append(s)
            else:
                resized_states.append(state)  # Non-LSTM states are unchanged

        return resized_states

    def get_one_hot_matrix(self, x_list):
        x_matrix = np.zeros([len(x_list), len(x_list[0]), self.corpus.vocab_size], np.float32)
        for i in range(len(x_list)):
            for j in range(len(x_list[i])):
                index = x_list[i][j]
                x_matrix[i, j, index] = 1
        return x_matrix

    def train_sequence(self, x_list, y_list, optimizer, learning_rate, batch_size=1, sequence_length=1):
        self.set_optimizer(optimizer, learning_rate)
        self.init_layer_states(batch_size)
        self.train()

        # creates x_list is a list of indexes
        x_list, y_list = Corpus.create_padded_sequence_list(x_list + [y_list[-1]],
                                                            sequence_length,
                                                            self.corpus.vocab_index_dict[self.unknown_token])

        if isinstance(self.layer_list[0], (torch.nn.LSTM, torch.nn.RNN, torch.nn.Linear)):
            x_list = self.get_one_hot_matrix(x_list)
            x_tensor = torch.tensor(x_list, dtype=torch.float32)
        elif isinstance(self.layer_list[0], torch.nn.Embedding):
            x_tensor = torch.tensor(x_list, dtype=torch.long)
        else:
            raise NotImplementedError(f"Unimplemented Layer Type {self.layer_list[0]}")

        y_tensor = torch.tensor(y_list, dtype=torch.long).view(-1)

        # Move tensors to the device
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        loss_sum = 0
        loss_n = 0
        start_time = time.time()
        for x_batch, y_batch in dataloader:
            current_batch_size = x_batch.size(0)
            if current_batch_size != batch_size:
                self.layer_state_list = self.resize_states(self.layer_state_list, current_batch_size)

            self.optimizer.zero_grad()
            output = self.forward(x_batch)
            loss = self.criterion(output, y_batch)
            loss_sum += loss.item()
            loss_n += len(x_batch)
            loss.backward()
            self.optimizer.step()

        took = time.time() - start_time
        perplexity = np.exp(loss_sum / loss_n)  # converting cross-entropy loss to perplexity score

        return took, perplexity

    def forward(self, x):
        for i in range(len(self.layer_list)):
            if self.layer_type_list[i] == 'lstm':
                h, c = self.layer_state_list[i]
                x, (h, c) = self.layer_list[i](x, (h, c))
                h = h.detach()
                c = c.detach()
                self.layer_state_list[i] = (h, c)
            elif self.layer_type_list[i] == 'srn':
                h = self.layer_state_list[i]
                x, h = self.layer_list[i](x, h)
                h = h.detach()
            elif self.layer_type_list[i] == 'gpt':
                pass
            else:
                x = self.layer_list[i](x)
        x = x[:, -1, :]
        return x

    def test_sequence(self, x_list, y_list, criterion, batch_size=1):
        self.init_layer_states(batch_size)
        self.eval()
        torch.no_grad()
        loss_sum = 0
        loss_n = 0
        start_time = time.time()
        for i in range(len(x_list)):
            output = self.forward(x_list[i])
            y = torch.tensor([y_list[i]]).to(self.device)
            loss = criterion(output, y)
            loss_sum += loss.item()
            loss_n += 1
        took = time.time() - start_time
        perplexity = self.calc_perplexity(loss_sum, loss_n)
        return took, perplexity

    def save(self, filename):
        """Save the instance to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """Load an instance from a file."""
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def get_layer_activation(self, layer):
        layer_activation = None
        if 0 > layer > (len(self.layer_list) - 1):
            raise ValueError(f"Layer {layer} doesnt exist in this model")
        return layer_activation

    def get_embedding_size(self, layer, unit_index):
        if len(self.layer_list) == 1:
            if layer != 0 and layer != -1:
                raise ValueError("Layer must be 0 or -1 if model has only one layer")
            else:
                embedding_size = len(self.layer_list[0].weight[unit_index, :])

        else:
            if layer == -1:
                if self.layer_type_list[-2] == 'embedding':
                    embedding_size = len(self.layer_list[-2].weight[:, unit_index])
                elif self.layer_type_list[-2] == 'linear':
                    embedding_size = len(self.layer_list[-2].weight[:, unit_index])
                elif self.layer_type_list[-2] == 'lstm':
                    embedding_size = int(
                        len(self.layer_list[-2].weight_ih_l0[:, unit_index]) / 4)  # TODO fix this to output weights
                elif self.layer_type_list[-2] == 'relu':
                    embedding_size = len(self.layer_list[-3].weight[:, unit_index])
                elif self.layer_type_list[-2] == 'srn':
                    embedding_size = len(self.layer_list[-2].weight[:, unit_index])
                else:
                    raise ValueError("ERROR: Unrecognized layer type", self.layer_type_list[layer])
            else:
                if self.layer_type_list[0] == 'embedding':
                    embedding_size = len(self.layer_list[layer].weight[unit_index, :])
                elif self.layer_type_list[layer] == 'linear':
                    embedding_size = len(self.layer_list[layer].weight[:, unit_index])
                elif self.layer_type_list[layer] == 'lstm':
                    embedding_size = len(self.layer_list[layer].weight_ih_l0[unit_index, :])
                else:
                    raise ValueError("ERROR: Unrecognized layer type", self.layer_type_list[layer])
        return embedding_size

    def get_weights(self, layer, unit_index=None):

        if len(self.layer_list) == 1 and (layer != 0 and layer != -1):
            raise ValueError("Layer must be 0 or -1 if model has only one layer")
        else:
            if layer == -1:
                if self.layer_type_list[layer] == 'embedding':
                    tensor = self.layer_list[layer].weight if unit_index is None else self.layer_list[layer].weight[
                                                                                      :, unit_index]
                elif self.layer_type_list[layer] == 'linear':
                    if unit_index is None:
                        tensor = self.layer_list[layer].weight
                    else:
                        tensor = self.layer_list[layer].weight[unit_index, :]
                elif self.layer_type_list[layer] == 'lstm' or self.layer_type_list[layer] == 'srn':
                    if unit_index is None:
                        tensor = self.layer_list[layer].weight_ih_l0
                    else:
                        tensor = self.layer_list[layer].weight_ih_l0[unit_index, :]


                elif self.layer_type_list[layer] == 'relu':
                    tensor = self.layer_list[layer - 1].weight if unit_index is None else self.layer_list[
                                                                                              layer - 1].weight[
                                                                                          unit_index, :]
                else:
                    raise ValueError("ERROR: Unrecognized layer type", self.layer_type_list[layer])
            else:
                if self.layer_type_list[0] == 'embedding':
                    tensor = self.layer_list[layer].weight if unit_index is None else self.layer_list[layer].weight[
                                                                                      unit_index, :]
                elif self.layer_type_list[layer] == 'linear':
                    tensor = self.layer_list[layer].weight if unit_index is None else self.layer_list[layer].weight[
                                                                                      :, unit_index]
                elif self.layer_type_list[layer] == 'lstm' or self.layer_type_list[layer] == 'srn':

                    tensor = self.layer_list[layer].weight_ih_l0
                    if unit_index is None:
                        tensor = tensor.t()
                    else:
                        tensor = tensor[:, unit_index]
                else:
                    raise ValueError("ERROR: Unrecognized layer type", self.layer_type_list[layer])

        if str(self.device.type) == 'cpu':
            weight_array = tensor.detach().numpy()
        elif self.device.type == 'cuda':
            weight_array = tensor.detach().cpu().numpy()  # Move tensor to CPU before converting
        elif str(self.device.type) == 'mps':
            weight_array = tensor.detach().to('cpu').numpy()
        else:
            raise ValueError("ERROR: Unrecognized device", self.device)

        return weight_array
