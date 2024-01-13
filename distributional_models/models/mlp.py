import time
import torch
import torch.nn as nn
from .neural_network import NeuralNetwork
from datetime import datetime


class MLP(NeuralNetwork):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 weight_init,
                 dropout_rate):

        super(MLP, self).__init__()
        self.model_name = None
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate

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

        self.layer_dict['relu'] = nn.ReLU()

        if self.dropout_rate > 0:
            self.layer_dict['dropout'] = nn.Dropout(self.dropout_rate)

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)

    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"mlp_{self.embedding_size}_{self.hidden_size}_{date_time_string}"

    def init_network(self):
        self.state_dict = {}

    def train_sequence(self, corpus, sequence, train_params):
        start_time = time.time()
        self.train()
        self.set_optimizer(train_params['optimizer'], train_params['learning_rate'], train_params['weight_decay'])
        self.set_criterion(train_params['criterion'])

        tokens_sum = 0
        loss_sum = 0

        sequence_length = 1

        x_batches, \
            single_y_batches, \
            y_window_batches = corpus.create_batched_sequence_lists(sequence,
                                                                    train_params['corpus_window_size'],
                                                                    train_params['batch_size'],
                                                                    sequence_length,
                                                                    self.device)

        y_batches = single_y_batches
        self.init_network()

        for batch_num, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            self.optimizer.zero_grad()
            output = self(x_batch)

            if train_params['l1_lambda']:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = self.criterion(output.view(-1, corpus.vocab_size),
                                      y_batch.view(-1)) + train_params['l1_lambda'] * l1_norm
            else:
                loss = self.criterion(output.view(-1, corpus.vocab_size), y_batch.view(-1))
            mask = y_batch.view(-1) != 0
            loss = (loss * mask).mean()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            tokens_sum += train_params['batch_size']

        loss_mean = loss_sum / tokens_sum
        took = time.time() - start_time

        return loss_mean, took

    def forward(self, x):

        embedding_out = self.layer_dict['embedding'](x)

        self.state_dict['zhidden'] = self.layer_dict['hidden'](embedding_out)
        self.state_dict['hidden'] = self.layer_dict['relu'](self.state_dict['zhidden'])

        if self.dropout_rate:
            self.state_dict['hidden'] = self.layer_dict['dropout'](self.state_dict['hidden'])

        # Output layer
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
