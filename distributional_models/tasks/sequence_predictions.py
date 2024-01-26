import copy
import numpy as np
import time


class SequencePredictions:

    def __init__(self, model, document_list, token_categories=None, target_categories=None, ignore_other=True):
        start_time = time.time()
        self.model = model
        self.document_list = document_list

        self.token_categories = token_categories
        self.target_categories = target_categories
        self.ignore_other = ignore_other

        self.target_category_freq_lists = None
        self.token_category_freq_dict = None
        self.target_category_freq_dict = None

        self.output_activation_sum_matrix = None
        self.output_activation_mean_matrix = None

        self.sequences_prediction_accuracy_mean_dict = None

        self.create_matrices()
        self.calculate_matrices()
        self.took = time.time() - start_time

    def create_matrices(self):
        vocab_size = len(self.model.vocab_index_dict)

        if self.token_categories is not None:
            if self.target_categories is not None:
                self.output_activation_sum_matrix = np.zeros([self.token_categories.num_categories,
                                                              self.target_categories.num_categories],
                                                             float)
            else:
                self.output_activation_sum_matrix = np.zeros([self.token_categories.num_categories, vocab_size], float)
        else:
            if self.target_categories is not None:
                self.output_activation_sum_matrix = np.zeros([vocab_size, self.target_categories.num_categories], float)
            else:
                self.output_activation_sum_matrix = np.zeros([vocab_size, vocab_size], float)

        self.output_activation_mean_matrix = copy.deepcopy(self.output_activation_sum_matrix)

    @staticmethod
    def get_index_list(target_category_list, target_category_index_dict):
        index_list = []
        for target in target_category_list:
            index_list.append(target_category_index_dict[target])
        return index_list

    def calculate_matrices(self):
        if self.token_categories is not None:
            token_category_index_dict = self.token_categories.instance_category_index_dict
        else:
            token_category_index_dict = self.model.vocab_index_dict

        if self.target_categories is not None:
            target_category_index_dict = self.target_categories.category_index_dict
        else:
            target_category_index_dict = self.model.vocab_index_dict

        for i, document in enumerate(self.document_list):
            for j, sequence in enumerate(document):
                output_activation_list, _ = self.model.test_sequence(sequence)

                for k, token in enumerate(sequence):
                    token_category_index = token_category_index_dict[token]
                    target_category_list = self.target_categories.document_category_lists[i][j][k]
                    target_category_index_list = self.get_index_list(target_category_list, target_category_index_dict)
                    target_category_freq_array = self.target_categories.category_freq_array_list[i][j][k]
                    output_array = output_activation_list[k]

                    if self.target_categories:
                        for m in range(self.model.vocab_size):
                            target_category_index = target_category_index_list[m]
                            self.output_activation_sum_matrix[
                                token_category_index, target_category_index] += output_array[m]
                            target_category_freq = target_category_freq_array[target_category_index]
                            self.output_activation_mean_matrix[
                                token_category_index, target_category_index] += output_array[m] / target_category_freq
                    else:
                        self.output_activation_sum_matrix[token_category_index, :] += output_array
                        self.output_activation_mean_matrix[token_category_index, :] += output_array / target_category_freq_array

        sum_matrix_row_sums = self.output_activation_sum_matrix.sum(1)
        safe_inverse = np.divide(1.0, sum_matrix_row_sums, where=sum_matrix_row_sums != 0, out=np.zeros_like(sum_matrix_row_sums, dtype=float))
        self.output_activation_sum_matrix = self.output_activation_sum_matrix * safe_inverse[:, np.newaxis]

        mean_matrix_row_sums = self.output_activation_mean_matrix.sum(1)
        safe_inverse = np.divide(1.0, mean_matrix_row_sums, where=mean_matrix_row_sums != 0, out=np.zeros_like(mean_matrix_row_sums, dtype=float))
        self.output_activation_mean_matrix = self.output_activation_mean_matrix * safe_inverse[:, np.newaxis]

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
