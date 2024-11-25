import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .neural_network import NeuralNetwork
import pandas as pd
from datetime import datetime
import csv


class SLRO(NeuralNetwork):
    def __init__(self,
                 vocab_list,
                 embedding_size,
                 weight_init,
                 dropout_rate,
                 act_func):

        super(SLRO, self).__init__(vocab_list)
        self.model_type = "slro"
        self.model_name = None
        self.embedding_size = embedding_size
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate
        self.activation_function = act_func

        self.define_network()
        self.create_model_name()

    def define_network(self):
        if self.embedding_size == 0:
            embedding_weights = torch.eye(self.vocab_size)
            self.layer_dict['embedding'] = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        else:
            self.layer_dict['embedding'] = nn.Embedding(self.vocab_size, self.embedding_size)
        if self.activation_function == 'relu':
            self.layer_dict['activation'] = nn.ReLU()
        elif self.activation_function == 'tanh':
            self.layer_dict['activation'] = nn.Tanh()
        elif self.activation_function == 'sigmoid':
            self.layer_dict['activation'] = nn.Sigmoid()

        if self.dropout_rate > 0:
            self.layer_dict['dropout'] = nn.Dropout(self.dropout_rate)

        self.layer_dict['output'] = nn.Linear(self.layer_dict['embedding'].embedding_dim, self.vocab_size)
        self.layer_dict['feedback'] = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"sl_{self.embedding_size}_{date_time_string}"

    def init_network(self, batch_size):
        self.state_dict = {}
        num_layers = 1
        self.state_dict['feedback'] = torch.zeros(num_layers, batch_size, self.vocab_size).to(self.device)

    def train_sequence(self, dataset, sequence, train_params, epoch=0):
        start_time = time.time()
        self.train()
        self.set_optimizer(train_params['optimizer'], train_params['learning_rate'],
                           train_params['weight_decay'], train_params['momentum'])
        self.set_criterion(train_params['criterion'])

        tokens_sum = 0
        loss_sum = 0

        sequence_length = 1

        x_batches, \
            single_y_batches, \
            y_window_batches = dataset.create_batched_sequence_lists(sequence,
                                                                    train_params['corpus_window_size'],
                                                                    train_params['corpus_window_direction'],
                                                                    train_params['batch_size'],
                                                                    sequence_length,
                                                                    self.device)

        y_batches = single_y_batches
        window_size = train_params['corpus_window_size']


        self.init_network(train_params['batch_size'])
        for batch_num, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            self.optimizer.zero_grad()
            output = self(x_batch)
            if train_params['l1_lambda']:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = self.criterion(output.view(-1, dataset.num_y,
                                      y_batch.view(-1)) + train_params['l1_lambda'] * l1_norm)
            else:
                loss = self.criterion(output.view(-1, dataset.num_y), y_batch.view(-1))
            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            tokens_sum += train_params['batch_size']

        loss_mean = loss_sum / tokens_sum
        took = time.time() - start_time
        return loss_mean, took

    def test_sequence(self, sequence, pad=True, softmax=True):
        self.eval()
        self.init_network(1)

        output_list = []
        hidden_state_list = []

        for token in sequence:
            outputs = self(torch.tensor([[self.vocab_index_dict[token]]])).detach()
            if softmax:
                outputs = F.softmax(outputs, dim=2).squeeze().numpy()
            else:
                outputs = outputs.squeeze().numpy()
            output_list.append(outputs)

        return output_list, output_list, output_list

    def forward(self, x):
        embedding_out = self.layer_dict['embedding'](x)
        feedback = self.layer_dict['feedback'](self.state_dict['feedback'])
        self.state_dict['output'] = self.layer_dict['output'](embedding_out) + feedback
        return self.state_dict['output']