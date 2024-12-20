import copy
import time
import torch
import torch.nn as nn
from .neural_network import NeuralNetwork
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import csv


class LSTM(NeuralNetwork):
    def __init__(self,
                 vocab_list,
                 embedding_size,
                 hidden_size,
                 weight_init,
                 dropout_rate,
                 act_func,
                 use_bias=True):

        super(LSTM, self).__init__(vocab_list)
        self.model_type = "lstm"
        self.model_name = None
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate
        self.activation_function = act_func
        self.use_bias = use_bias

        self.define_network()
        self.create_model_name()

    def define_network(self):

        if self.embedding_size == 0:
            embedding_weights = torch.eye(self.vocab_size)
            self.layer_dict['embedding'] = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            self.layer_dict['lstm'] = nn.LSTM(self.vocab_size, self.hidden_size, dropout=self.dropout_rate,
                                              batch_first=True)
        else:
            self.layer_dict['embedding'] = nn.Embedding(self.vocab_size, self.embedding_size)
            self.layer_dict['lstm'] = nn.LSTM(self.embedding_size, self.hidden_size, dropout=self.dropout_rate,
                                              batch_first=True, nonlinearity=self.activation_function)

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)

    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"lstm_{self.embedding_size}_{self.hidden_size}_{date_time_string}"

    def init_network(self, batch_size):

        self.state_dict = {}
        num_layers = 1
        self.state_dict['hidden'] = (torch.zeros(num_layers, batch_size, self.hidden_size).to(self.device),
                                     torch.zeros(num_layers, batch_size, self.hidden_size).to(self.device))

    def train_sequence(self, dataset, sequence, train_params, save_example_corpus=False):
        self.epoch += 1
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
            y_window_batches = dataset.create_batched_sequence_lists(sequence,
                                                                    corpus_window_size,
                                                                    train_params['corpus_window_direction'],
                                                                    train_params['batch_size'],
                                                                    train_params['sequence_length'],
                                                                    self.device)

        y_batches = single_y_batches
        if save_example_corpus:
            if train_params['sequence_length'] == 1:
                input_list = [self.vocab_list[t.numpy().item()] for t in x_batches]
                output_list = [self.vocab_list[t.numpy().item()] for t in y_batches]

            else:
                input_list = [self.vocab_list[index] for t in x_batches for index in t.numpy().flatten()]
                output_list = [self.vocab_list[t.numpy().item()] for t in y_batches]

            sequence_length = train_params['sequence_length']  # Adjust this parameter as needed

            # Generate pairs
            pairs = []
            for i in range(0, len(input_list), sequence_length):
                input_tokens = input_list[i:i + sequence_length]
                output_token = output_list[i // sequence_length]
                pairs.append(input_tokens + [output_token])

            # Define the CSV file name
            csv_file_name = \
                '/Users/jingfengzhang/FirstYearProject/AyB/ayb/corpus and co_occurrence/lstm_training_data.csv'

            # Write the data into the CSV file
            with open(csv_file_name, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)

                # Write the header dynamically based on sequence length
                header = [f"input{i + 1}" for i in range(sequence_length)] + ["output"]
                writer.writerow(header)

                # Write the rows
                writer.writerows(pairs)

            print(f"Data successfully written to {csv_file_name}")

        self.init_network(train_params['batch_size'])

        for batch_num, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            self.optimizer.zero_grad()
            output = self(x_batch)
            self.state_dict['hidden'] = (self.state_dict['hidden'][0].detach(),
                                         self.state_dict['hidden'][1].detach())

            if train_params['l1_lambda']:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = self.criterion(output.view(-1, dataset.num_y),
                                      y_batch.view(-1)) + train_params['l1_lambda'] * l1_norm
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

        np.set_printoptions(formatter={'float': '{:0.3f}'.format}, linewidth=np.inf)
        # print(f"{correct_sum/len(x_batches):0.3f}, {incorrect_sum/len(x_batches):0.3f}")

        return loss_mean, took

    def test_sequence(self, sequence, pad=False, softmax=True):
        self.eval()
        self.init_network(1)
        previous_state_list = []
        output_list = []
        hidden_state_list = []
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
                sequence_list.append(padded_sequence[i:i+sequence_length])
        for sequence in sequence_list:
            previous_state_list.append(self.state_dict['hidden'][0].squeeze().numpy())
            outputs = self(torch.tensor([sequence])).detach()
            self.state_dict['hidden'] = (self.state_dict['hidden'][0].detach(),
                                         self.state_dict['hidden'][1].detach())
            hidden_state_list.append(copy.deepcopy(self.state_dict['hidden'][0].squeeze().numpy()))

            if softmax:
                outputs = F.softmax(outputs, dim=1).squeeze().numpy()
            else:
                outputs = outputs.squeeze().numpy()
            output_list.append(outputs)
        return previous_state_list, output_list, hidden_state_list

    def forward(self, x):
        embedding_out = self.layer_dict['embedding'](x)
        # LSTM layer
        lstm_out, self.state_dict['hidden'] = self.layer_dict['lstm'](embedding_out, self.state_dict['hidden'])

        # Only take the output from the final timestep
        # You can modify this part to return the output at each timestep
        lstm_out = lstm_out[:, -1, :]

        # Output layer
        o = self.layer_dict['output'](lstm_out)

        return o

    def get_states(self, x, layer):

        if layer in ['hidden', 'cell', 'combined', 'output']:
            o = self(x)  # [1,5,vocab_size]
            state = self.state_dict["hidden"]
            if layer == 'hidden':
                state = state[0]
            elif layer == 'cell':
                state = state[1]
            elif layer == 'combined':
                state = torch.cat((state[0], state[1]), dim=2)
                # TODO check to make sure this concatenates them the right way
                # [batch_size, seq_len, hidden_size] --> [batch_size, seq_len, hidden_size*2]
            elif state == 'output':
                state = o
        else:
            raise ValueError(f"Improper layer request {layer} for LSTM")

        return state
