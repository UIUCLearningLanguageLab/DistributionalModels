import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .neural_network import NeuralNetwork
import pandas as pd
from datetime import datetime
import csv


class MLR(NeuralNetwork):
    def __init__(self,
                 vocab_list,
                 embedding_size,
                 add_on,
                 hidden_size,
                 weight_init,
                 dropout_rate,
                 act_func):

        super(MLR, self).__init__(vocab_list)
        self.model_type = "mlr"
        self.model_name = None
        self.embedding_size = embedding_size
        self.add_on = add_on
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate
        self.activation_function = act_func

        self.define_network()
        self.create_model_name()

    def define_network(self):
        if self.embedding_size == 0:
            embedding_weights = torch.eye(self.vocab_size)
            self.layer_dict['embedding'] = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            self.layer_dict['hidden'] = nn.Linear(self.vocab_size, self.hidden_size)
        else:
            self.layer_dict['embedding'] = nn.Embedding(self.vocab_size, self.embedding_size)
            self.layer_dict['hidden'] = nn.Linear(self.embedding_size, self.hidden_size)
        if self.activation_function == 'relu':
            self.layer_dict['activation'] = nn.ReLU()
        elif self.activation_function == 'tanh':
            self.layer_dict['activation'] = nn.Tanh()

        if self.dropout_rate > 0:
            self.layer_dict['dropout'] = nn.Dropout(self.dropout_rate)

        self.layer_dict['feedback'] = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.layer_dict['recurrent'] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)


    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"mlro{self.add_on}_{self.embedding_size}_{self.hidden_size}_{date_time_string}"

    def init_network(self, batch_size):
        self.state_dict = {}
        num_layers = 1
        self.state_dict['feedback'] = torch.zeros(batch_size, self.vocab_size).to(self.device)
        self.state_dict['hidden'] = torch.zeros(batch_size, self.hidden_size).to(self.device)

    def train_sequence(self, dataset, sequence, train_params):
        start_time = time.time()
        self.train()
        self.set_optimizer(train_params['optimizer'], train_params['learning_rate'],
                           train_params['weight_decay'], train_params['momentum'])
        self.set_criterion(train_params['criterion'])

        tokens_sum = 0
        loss_sum = 0

        corpus_window_size = 1  # this is for creating w2v style windowed pairs in the dataset
        x_batches, \
        single_y_batches, \
        y_window_batches, = dataset.create_batched_sequence_lists(sequence,
                                                                  corpus_window_size,
                                                                  train_params['corpus_window_direction'],
                                                                  train_params['batch_size'],
                                                                  train_params['sequence_length'],
                                                                  self.device)

        y_batches = single_y_batches

        self.init_network(train_params['batch_size'])
        for batch_num, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            self.init_network(train_params['batch_size'])
            self.optimizer.zero_grad()
            output = self(x_batch)
            if train_params['l1_lambda']:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = self.criterion(output.view(-1, dataset.num_y),
                                      y_batch.view(-1)) + train_params['l1_lambda'] * l1_norm
            else:
                loss = self.criterion(output.view(-1, dataset.num_y), y_batch.view(-1))
            self.state_dict['hidden'] = self.state_dict['hidden'].detach()
            self.state_dict['feedback'] = output.detach()

            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            tokens_sum += train_params['batch_size']

        loss_mean = loss_sum / tokens_sum
        took = time.time() - start_time

        return loss_mean, took

    def test_sequence(self, sequence, pad=False, softmax=True):
        self.eval()
        self.init_network(1)

        previous_state_list = []
        hidden_state_list = []
        output_list = []
        sequence_length = self.params['sequence_length']
        # pad sequence based on sequence length
        sequence = [self.vocab_index_dict[token] for token in sequence]
        if pad:
            padded_sequence = [self.vocab_index_dict[self.params['unknown_token']]] * (sequence_length - 1) + sequence
        else:
            padded_sequence = sequence
        sequence_list = []
        for i in range(len(padded_sequence)):
            if i + sequence_length <= len(padded_sequence):
                sequence_list.append(padded_sequence[i:i + sequence_length])
        for sequence in sequence_list:
            previous_state_list.append(self.state_dict['hidden'].squeeze().numpy())
            outputs = self(torch.tensor([sequence])).detach()

            self.state_dict['hidden'] = self.state_dict['hidden'].detach()
            self.state_dict['feedback'] = outputs
            hidden_state_list.append(self.state_dict['hidden'].squeeze().numpy())

            if softmax:
                outputs = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
            else:
                outputs = outputs.squeeze().numpy()

            output_list.append(outputs)

        return previous_state_list, output_list, hidden_state_list

    def forward(self, x):
        batch_size, seq_length = x.shape
        mlro_outs = []
        for t in range(seq_length):
            x_t = x[:, t]
            embedding_out = self.layer_dict['embedding'](x_t)
            zhidden = self.layer_dict['hidden'](embedding_out)
            if self.add_on == 'oh':
                hidden = self.layer_dict['activation'](zhidden + self.layer_dict['feedback'](self.state_dict['feedback'])
                                                       + self.layer_dict['recurrent'](self.state_dict['hidden']))
            elif self.add_on == 'o':
                hidden = self.layer_dict['activation'](
                    zhidden + self.layer_dict['feedback'](self.state_dict['feedback']))
            elif self.add_on == 'h':
                hidden = self.layer_dict['activation'](
                    zhidden + self.layer_dict['recurrent'](self.state_dict['hidden']))
            if self.dropout_rate:
                hidden = self.layer_dict['dropout'](hidden)
            self.state_dict['hidden'] = hidden
            mlro_outs.append(hidden)
            # Output layer
        mlro_outs = torch.stack(mlro_outs, dim=1)
        # self.state_dict['hidden'] = mlro_outs[:, -1, :]
        self.state_dict['output'] = self.layer_dict['output'](self.state_dict['hidden'])
        return self.state_dict['output']

    def get_states(self, x, layer):
        o = self(x)  # [1,5,vocab_size]
        if layer in ['zhidden', 'hidden']:
            state = self.state_dict[layer]
        elif layer == 'output':
            state = o
        else:
            raise ValueError(f"Improper layer request {layer} for MLP")
        return state
