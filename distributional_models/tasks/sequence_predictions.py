import copy

import numpy as np
import time
import torch.nn.functional as F
import torch
np.set_printoptions(formatter={'float': '{:0.3f}'.format}, linewidth=np.inf)


class SequencePredictions:

    def __init__(self, model, document_list, sequence_target_label_list=None, token_category_dict=None,
                 target_category_index_dict=None,
                 ignore_other=True):
        start_time = time.time()
        self.model = model
        self.document_list = document_list

        self.token_category_dict = token_category_dict
        self.target_category_index_dict = target_category_index_dict
        self.ignore_other = ignore_other
        self.token_category_list = None
        self.token_category_index_dict = None
        self.num_token_categories = None

        self.sequence_target_label_list = sequence_target_label_list
        self.target_category_list = None
        self.num_target_categories = None

        self.token_vocab_index_lists = None
        self.token_category_index_lists = None
        self.target_category_index_lists = None
        self.target_category_freq_lists = None
        self.token_category_freq_dict = None
        self.target_category_freq_dict = None

        self.output_activation_sum_matrix = None
        self.output_activation_mean_matrix = None

        self.sequences_prediction_accuracy_mean_dict = None
        self.sequence_prediction_accuracy_mean_array = []  # accuracy for each category
        self.sequence_prediction_accuracy_mean = 100.0  # overall accuracy

        self.create_token_information()
        self.create_target_information()
        self.create_index_lists()
        self.create_matrices()
        self.calculate_matrices()
        self.took = time.time() - start_time

    def create_matrices(self):
        vocab_size = len(self.model.vocab_index_dict)

        if self.token_category_dict:
            if self.sequence_target_label_list:
                self.output_activation_sum_matrix = np.zeros([self.num_token_categories, self.num_target_categories],
                                                             float)
            else:
                self.output_activation_sum_matrix = np.zeros([self.num_token_categories, vocab_size], float)
        else:
            if self.sequence_target_label_list:
                self.output_activation_sum_matrix = np.zeros([vocab_size, self.num_target_categories], float)
            else:
                self.output_activation_sum_matrix = np.zeros([vocab_size, vocab_size], float)

        self.output_activation_mean_matrix = copy.deepcopy(self.output_activation_sum_matrix)

    def create_index_lists(self):
        for i, document in enumerate(self.document_list):
            for j, sequence in enumerate(document):
                token_vocab_index_list = []
                token_category_index_list = []
                target_category_index_list = []
                target_category_freq_list = []

                for k, token in enumerate(sequence):
                    token_vocab_index_list.append(self.model.vocab_index_dict[token])
                    if self.token_category_dict:
                        token_category_index_list.append(self.token_category_index_dict[self.token_category_dict[token]])
                    else:
                        token_category_index_list.append(self.model.vocab_index_dict[token])

                    output_category_index_list = []

                    target_category_freq_array = np.zeros([self.num_target_categories])
                    for l in range(self.model.vocab_size):
                        if self.sequence_target_label_list:
                            output_category_index = self.target_category_index_dict[self.sequence_target_label_list[i][j][k][l]]
                        else:
                            output_category_index = k
                        output_category_index_list.append(output_category_index)
                        target_category_freq_array[output_category_index] += 1

                    target_category_index_list.append(output_category_index_list)
                    target_category_freq_list.append(target_category_freq_array)

                self.token_vocab_index_lists.append(token_vocab_index_list)
                self.token_category_index_lists.append(token_category_index_list)
                self.target_category_index_lists.append(target_category_index_list)
                self.target_category_freq_lists.append(target_category_freq_list)

    def create_token_information(self):
        self.token_category_index_lists = []
        self.token_vocab_index_lists = []
        if self.token_category_dict:
            self.token_category_list = sorted(list(set(self.token_category_dict.values())))
            self.token_category_index_dict = {value: index for index, value in enumerate(self.token_category_list)}
            self.num_token_categories = len(self.token_category_list)

        self.token_category_freq_dict = {key: 0 for key in self.token_category_dict}

    def create_target_information(self):
        self.target_category_index_lists = []
        self.target_category_freq_lists = []
        if self.sequence_target_label_list:
            if self.target_category_index_dict is None:
                self.target_category_list = list({item for sublist1 in self.sequence_target_label_list
                                          for sublist2 in sublist1
                                          for sublist3 in sublist2
                                          for item in sublist3})
                self.target_category_index_dict = {value: index for index, value in enumerate(self.target_category_list)}
                self.num_target_categories = len(self.target_category_list)
            else:
                self.target_category_list = list(self.target_category_index_dict.keys())
                self.num_target_categories = len(self.target_category_index_dict)

    def calculate_matrices(self):
        self.model.eval()
        self.model.init_network(1)

        for document in self.document_list:
            for i, sequence in enumerate(document):
                for j, token in enumerate(sequence):
                    token_vocab_index = self.token_vocab_index_lists[i][j]
                    token_category_index = self.token_category_index_lists[i][j]
                    target_category_index_array = self.target_category_index_lists[i][j]
                    target_category_freq_array = self.target_category_freq_lists[i][j]
                    raw_outputs = self.model(torch.tensor([[token_vocab_index]])).detach()
                    self.model.state_dict['hidden'] = (self.model.state_dict['hidden'][0].detach(),
                                                       self.model.state_dict['hidden'][1].detach())
                    probs = F.softmax(raw_outputs, dim=1).squeeze().numpy()

                    self.token_category_freq_dict[token] += 1

                    if self.sequence_target_label_list:
                        for k, prob in enumerate(probs):
                            target_category_index = target_category_index_array[k]
                            self.output_activation_sum_matrix[token_category_index, target_category_index] += prob
                            target_category_freq = target_category_freq_array[target_category_index]

                            self.output_activation_mean_matrix[
                                token_category_index, target_category_index] += prob / target_category_freq
                    else:
                        self.output_activation_sum_matrix[token_category_index, :] += probs
                        self.output_activation_mean_matrix[token_category_index, :] += probs / target_category_freq_array

        sum_matrix_row_sums = self.output_activation_sum_matrix.sum(1)
        safe_inverse = np.divide(1.0, sum_matrix_row_sums, where=sum_matrix_row_sums != 0, out=np.zeros_like(sum_matrix_row_sums, dtype=float))
        self.output_activation_sum_matrix = self.output_activation_sum_matrix * safe_inverse[:, np.newaxis]

        mean_matrix_row_sums = self.output_activation_mean_matrix.sum(1)
        safe_inverse = np.divide(1.0, mean_matrix_row_sums, where=mean_matrix_row_sums != 0, out=np.zeros_like(mean_matrix_row_sums, dtype=float))
        self.output_activation_mean_matrix = self.output_activation_mean_matrix * safe_inverse[:, np.newaxis]

        self.print_matrix(self.output_activation_sum_matrix, self.token_category_list, self.target_category_list)
        self.print_matrix(self.output_activation_mean_matrix, self.token_category_list, self.target_category_list)

    @staticmethod
    def print_matrix(matrix, row_labels, column_labels):
        # Determine the width of each column
        col_width = 8

        # Print the column labels
        print(" " * 10, end=" ")  # Space for row labels
        for label in column_labels:
            print(f"{label:>{col_width}}", end=" ")
        print()

        # Print the matrix rows with row labels
        for label, row in zip(row_labels, matrix):
            print(f"{label:>{10}}", end=" ")  # Right-align the row label in 10 spaces
            for cell in row:
                print(f"{cell:>{col_width}.3f}", end=" ")  # Right-align and format the cell value
            print()
