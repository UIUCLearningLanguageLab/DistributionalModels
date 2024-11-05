import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .neural_network import NeuralNetwork
import pandas as pd
from datetime import datetime
import csv


class MLP(NeuralNetwork):
    def __init__(self,
                 vocab_list,
                 embedding_size,
                 hidden_size,
                 weight_init,
                 dropout_rate,
                 act_func):

        super(MLP, self).__init__(vocab_list)
        self.model_type = "mlp"
        self.model_name = None
        self.embedding_size = embedding_size
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

        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)

    def create_model_name(self):
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"mlp_{self.embedding_size}_{self.hidden_size}_{date_time_string}"

    def init_network(self):
        self.state_dict = {}

    def train_sequence(self, dataset, sequence, train_params, epoch=0,
                       save_example_corpus=False):
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
        if save_example_corpus:
            corpus_save_path = '/Users/jingfengzhang/FirstYearProject/AyB/ayb/corpus and co_occurrence/'
            input_list = [self.vocab_list[t.numpy().item()] for t in x_batches]
            output_list = [self.vocab_list[t.numpy().item()] for t in y_batches]
            pairs = []
            if train_params['corpus_window_direction'] == 'forward':
                csv_file_name = corpus_save_path + f'forward_w2v_training_data_{epoch}.csv'
                headers = ["input"] + [f"output{i + 1}" for i in range(train_params['corpus_window_size'])]
                for i in range(0, len(input_list), window_size):
                    input_token = input_list[i]
                    output_tokens = output_list[i:i + window_size]
                    pairs.append([input_token] + output_tokens)
            elif train_params['corpus_window_direction'] == 'backward':
                csv_file_name = corpus_save_path + 'backward_w2v_training_data.csv'
                headers = [f"input{i + 1}" for i in range(train_params['corpus_window_size'])] + ["output"]
                for i in range(0, len(output_list), window_size):
                    output_token = output_list[i]
                    input_tokens = input_list[i:i + window_size]
                    pairs.append(input_tokens + [output_token])

            # Write the data into the CSV file
            with open(csv_file_name, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)

                # Write the header
                writer.writerow(headers)

                # Write the rows
                writer.writerows(pairs)

            # Create a DataFrame from the pairs
            df = pd.DataFrame(pairs, columns=headers)

            # Melt the DataFrame to create a long format DataFrame with input-output pairs
            if train_params['corpus_window_direction'] == 'forward':
                melted_df = df.melt(id_vars=headers[:1], value_vars=headers[1:],
                                    var_name='output_type', value_name='output')
                co_occurrence_table = melted_df.groupby(['input', 'output']).size().unstack(fill_value=0)
                co_occurrence_table.to_csv(corpus_save_path + 'forward_co_occurrence_table.csv')
            elif train_params['corpus_window_direction'] == 'backward':
                melted_df = df.melt(id_vars=headers[-1:], value_vars=headers[:-1],
                                    var_name='input_type', value_name='input')
                # Create the co-occurrence table by counting the frequency of each input-output pair
                co_occurrence_table = melted_df.groupby(['output', 'input']).size().unstack(fill_value=0)
                co_occurrence_table.to_csv(corpus_save_path + 'backward_co_occurrence_table.csv')

        self.init_network()
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
        if save_example_corpus:
            return loss_mean, took, co_occurrence_table
        else:
            return loss_mean, took

    def test_sequence(self, sequence, pad=True, softmax=True):
        self.eval()
        self.init_network()

        output_list = []
        hidden_state_list = []

        for token in sequence:
            outputs = self(torch.tensor([[self.vocab_index_dict[token]]])).detach()
            hidden_state_list.append(copy.deepcopy(self.state_dict['hidden'].detach().squeeze().numpy()))
            if softmax:
                outputs = F.softmax(outputs, dim=2).squeeze().numpy()
            else:
                outputs = outputs.squeeze().numpy()
            output_list.append(outputs)

        return hidden_state_list, output_list, hidden_state_list

    def forward(self, x):

        embedding_out = self.layer_dict['embedding'](x)

        self.state_dict['zhidden'] = self.layer_dict['hidden'](embedding_out)
        # self.state_dict['hidden'] = self.layer_dict['relu'](self.state_dict['zhidden'])
        self.state_dict['hidden'] = self.layer_dict['activation'](self.state_dict['zhidden'])
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
