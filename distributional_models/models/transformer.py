import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.nn import functional
from .neural_network import NeuralNetwork
from datetime import datetime


class Transformer(NeuralNetwork):
    def __init__(self,
                 vocab_list,
                 sequence_length,
                 embedding_size,
                 num_heads,
                 attention_size,
                 hidden_size,
                 weight_init,
                 device):

        super(Transformer, self).__init__(vocab_list)
        self.model_type = "transformer"
        self.model_name = None
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.attention_size = attention_size
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.device = device

        self.define_network()
        self.create_model_name()
        self.init_network()

    def init_network(self):
        self.state_dict = {}

    def define_network(self):
        self.layer_dict['token_embeddings_table'] = nn.Embedding(self.vocab_size, self.embedding_size)
        self.layer_dict['position_embeddings_table'] = nn.Embedding(self.sequence_length, self.embedding_size)
        self.layer_dict['combined_input_module'] = CombinedInput(self.layer_dict['token_embeddings_table'],
                                                                 self.layer_dict['position_embeddings_table'],
                                                                 self.device)
        self.layer_dict['attention_weighted_values'] = MultiHeadAttention(self.num_heads,
                                                                          self.attention_size // self.num_heads,
                                                                          self.embedding_size,
                                                                          self.sequence_length)
        self.layer_dict['hidden_layer'] = FeedForward(self.attention_size, self.hidden_size)
        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)

    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = \
            f"tf_{self.embedding_size}_{self.num_heads}_{self.attention_size}_{self.hidden_size}_{date_time_string}"

    def train_sequence(self, corpus, sequence, train_params):
        start_time = time.time()
        self.train()
        self.set_optimizer(train_params['optimizer'], train_params['learning_rate'], train_params['weight_decay'])
        self.set_criterion(train_params['criterion'])

        tokens_sum = 0
        loss_sum = 0

        corpus_window_size = 1  # this is for creating w2v style windowed pairs in the dataset

        x_batches, single_y_batches, y_window_batches = corpus.create_batched_sequence_lists(sequence,
                                                                                             corpus_window_size,
                                                                                             train_params['batch_size'],
                                                                                             self.sequence_length,
                                                                                             self.device)

        if train_params['transformer_target_output'] == 'y_window':
            y_batches = y_window_batches
            single_y = False
        elif train_params['transformer_target_output'] == 'single_y':
            y_batches = single_y_batches
            single_y = True
        elif train_params['transformer_target_output'] == 'x_window':
            y_batches = x_batches
            single_y = False
        else:
            raise ValueError(f"Unrecognized transformer target output {train_params['transformer_target_output']}")

        for x_batch, y_batch in zip(x_batches, y_batches):

            self.optimizer.zero_grad()
            output = self(x_batch)
            if single_y:
                output = output[:, -1, :]

            else:
                B, T, C = output.shape
                output = output.view(B * T, C)

            if train_params['l1_lambda']:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = self.criterion(output, y_batch.view(-1)) + train_params['l1_lambda'] * l1_norm
            else:
                loss = self.criterion(output, y_batch.view(-1))

            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            tokens_sum += train_params['batch_size']

        loss_mean = loss_sum / tokens_sum
        took = time.time() - start_time

        return loss_mean, took

    def test_sequence(self, sequence, softmax=True):
        self.eval()
        self.init_network()

        output_list = []
        hidden_state_list = []
        pads = ['<unk>' for _ in range(self.sequence_length-1)]
        padded_sequence = pads+sequence
        sequence = [self.vocab_index_dict[token] for token in padded_sequence]
        x_batches = [sequence[i:i+self.sequence_length] for i in range(len(sequence)-self.sequence_length+1)]
        for x_batch in x_batches:
            outputs = self(torch.tensor([x_batch]))[:, -1, :].detach()
            hidden_state_list.append(self.state_dict['hidden'][:, -1, :].unsqueeze(1).detach())
            if softmax:
                outputs = F.softmax(outputs, dim=1).squeeze().numpy()
            output_list.append(outputs)

        return output_list, hidden_state_list

    def forward(self, x_window):
        _, sequence_length = x_window.shape
        combined_input = self.layer_dict['combined_input_module'](x_window, sequence_length)
        attention_weighted_values, attention_weights = self.layer_dict['attention_weighted_values'](combined_input)
        h = self.layer_dict['hidden_layer'](attention_weighted_values)
        outputs = self.layer_dict['output'](h)  # (Batch*Time*vocab_size)

        self.state_dict['combined_input'] = combined_input
        self.state_dict['attention_weighted_values'] = attention_weighted_values
        self.state_dict['attention_weights'] = attention_weights
        self.state_dict['hidden'] = h

        return outputs

    def get_states(self, x, layer):
        o = self(x)  # [1,5,vocab_size]
        if layer in ['combined_input', 'attention_weighted_values', 'attention_weights', 'h']:
            state = self.state_dict[layer]
            # TODO if, like LSTM, these need to be further processed or given further options, do that here
        elif layer == 'output':
            state = o
        else:
            raise ValueError(f"Improper layer request {layer} for Transformer")
        return state


class CombinedInput(nn.Module):
    def __init__(self, token_embeddings_table, position_embeddings_table, device):
        super().__init__()
        self.token_embeddings_table = token_embeddings_table
        self.position_embeddings_table = position_embeddings_table
        self.device = device

    def forward(self, idx, T):
        token_embed = self.token_embeddings_table(idx)
        position_embed = self.position_embeddings_table(torch.arange(T, device=self.device))
        combined_input = token_embed + position_embed
        return combined_input


class Head(nn.Module):
    def __init__(self, head_size, embed_size, sequence_length):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_mask_attention_weights = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B ,T, C)
        q = self.query(x)  # (B ,T, C)
        v = self.value(x)  # (B ,T, C)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
        self.pre_mask_attention_weights = wei.clone()
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = functional.softmax(wei, dim=1)  # (B, T, T)
        out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, sequence_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, sequence_length) for i in range(num_heads)])

    def forward(self, x):
        attention_weights_list = []
        head_outputs = []
        for head in self.heads:
            head_output = head(x)
            head_outputs.append(head_output)

            # Assuming pre_mask_attention_weights are stored in each head
            attention_weights_list.append(head.pre_mask_attention_weights)

        # Concatenating the outputs from all heads
        combined_output = torch.cat(head_outputs, dim=-1)

        # Average the attention weights across all heads
        combined_attention_weights = torch.cat(attention_weights_list, dim=-1)
        return combined_output, combined_attention_weights


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, hidden_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)
