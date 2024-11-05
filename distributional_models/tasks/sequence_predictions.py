import copy
import numpy as np
import time
import pandas as pd
from collections import defaultdict


class SequencePredictions:

    def __init__(self, model, corpus, params, document_list, token_list=None, target_list=None,
                 token_categories=None, target_categories=None):
        start_time = time.time()
        self.model = model
        self.corpus = corpus
        self.document_list = document_list
        self.params = params
        self.params['unknown_token'] = '<unk>'
        self.accuracy_dict = {}

        if token_list is None:
            self.token_list = copy.deepcopy(model.vocab_list)
        else:
            self.token_list = token_list
        if target_list is None:
            self.target_list = copy.deepcopy(model.vocab_list)
        else:
            self.target_list = target_list

        self.token_categories = token_categories
        self.token_category_list = None
        self.token_category_index_dict = None
        self.num_token_categories = None

        self.target_categories = target_categories
        self.target_category_list = None
        self.target_category_index_list = None
        self.num_target_categories = None

        self.target_category_freq_lists = None
        self.token_category_freq_dict = None
        self.target_category_freq_dict = None

        self.output_activation_sum_matrix = None
        self.output_activation_mean_matrix = None

        self.sequences_prediction_accuracy_mean_dict = None

        self.get_token_categories()

        if self.target_categories is not None:
            self.get_target_categories()
        else:
            self.target_category_list = copy.deepcopy(self.model.vocab_list)
            self.target_category_index_dict = copy.deepcopy(self.model.vocab_index_dict)
            self.num_target_categories = len(self.target_category_list)

        self.calculate_matrices()
        self.took = time.time() - start_time

    def get_token_categories(self):

        if self.token_categories is None:
            self.token_category_list = copy.deepcopy(self.model.vocab_list)
            self.token_category_index_dict = copy.deepcopy(self.model.vocab_index_dict)
            self.num_token_categories = len(self.token_category_list)

        else:
            self.token_category_list = []
            self.token_category_index_dict = {}

            for token in self.token_list:
                if token not in self.model.vocab_index_dict:
                    raise Exception(f"token {token} not in vocab_index_dict")
                else:
                    if token not in self.token_categories.instance_category_dict:
                        raise Exception(f"token {token} not in categories")
                    else:
                        category = self.token_categories.instance_category_dict[token]
                        if category not in self.token_category_list:
                            self.token_category_list.append(category)

            self.token_category_list.sort()
            self.num_token_categories = len(self.token_category_list)
            for i in range(self.num_token_categories):
                self.token_category_index_dict[self.token_category_list[i]] = i

    def get_target_categories(self):
        self.target_category_list = []
        self.target_category_index_dict = {}

        for target in self.target_list:
            if target not in self.model.vocab_index_dict:
                raise Exception(f"target {target} not in vocab_index_dict")

        for i, document in enumerate(self.document_list):
            for j, sequence in enumerate(document):
                for k, token in enumerate(sequence):
                    for l in range(self.model.vocab_size):
                        current_output = self.model.vocab_list[l]
                        if current_output in self.target_list:
                            #print(i, j, k, l)
                            current_category = self.target_categories.document_category_lists[i][j][k][l]
                            if current_category not in self.target_category_list:
                                self.target_category_list.append(current_category)

        self.target_category_list.sort()
        self.num_target_categories = len(self.target_category_list)
        for i in range(self.num_target_categories):
            self.target_category_index_dict[self.target_category_list[i]] = i

    def calculate_matrices(self):
        self.output_activation_sum_matrix = np.zeros([self.num_token_categories, self.num_target_categories], float)
        self.output_activation_mean_matrix = np.zeros([self.num_token_categories, self.num_target_categories], float)
        correct_count_dict = {'A_group_correct': 0,
                              'A_subgroup_correct': 0,
                              'A_alt_subgroup_correct': 0,
                              'A_omit_correct': 0,
                              'B_group_correct': 0,
                              'B_subgroup_correct': 0,
                              'B_omit_correct': 0,
                              'B_alt_subgroup_correct': 0,
                              'y_group_correct': 0
                              }

        model_analysis_dict = defaultdict(list)
        prediction_count_dict = {target_category: 0 for target_category in self.target_categories.document_category_lists[0][0][0]}
        for i, document in enumerate(self.document_list):
            output_activation_matrix = np.zeros([len(document), 4, self.model.vocab_size])
            for j, sequence in enumerate(document):
                self.model.params = self.params
                _, output_activation_list, _ = self.model.test_sequence(sequence, pad=True)
                for k, token in enumerate(sequence):
                    token_category = self.token_categories.instance_category_dict[token]
                    output_array = output_activation_list[k]
                    output_activation_matrix[j, k] = output_array
                    prediction = np.argmax(output_array)
                    target_category_list = self.target_categories.document_category_lists[i][j][k]
                    y_indices = [index for index, x in enumerate(target_category_list) if
                                       x == 'y']
                    period_indices = [index for index, x in enumerate(target_category_list) if
                                 x == '.']
                    b_legal_indices = [index for index, x in enumerate(target_category_list) if
                                       x == 'B_Legal' or x == 'B_Present']
                    b_current_indices = [index for index, x in enumerate(target_category_list) if x == 'B_Present']
                    b_illegal_indices = [index for index, x in enumerate(target_category_list) if
                                         x == 'B_Illegal']
                    b_omit_indices = [index for index, x in enumerate(target_category_list) if x == 'B_Omitted']
                    a_legal_indices = [index for index, x in enumerate(target_category_list) if
                                       x == 'A_Legal' or x == 'A_Present']
                    a_current_indices = [index for index, x in enumerate(target_category_list) if x == 'A_Present']
                    a_illegal_indices = [index for index, x in enumerate(target_category_list) if
                                         x == 'A_Illegal']
                    a_omit_indices = [index for index, x in enumerate(target_category_list) if
                                      x == 'A_Omitted']
                    if token in self.token_list:
                        model_analysis_dict['current_B'].append(output_array[b_current_indices])
                        b_legal_only_indices = list(set(b_legal_indices) - set(b_current_indices))
                        model_analysis_dict['legal_B'].append(output_array[b_legal_only_indices])
                        model_analysis_dict['omit_B'].append(output_array[b_omit_indices])
                        model_analysis_dict['illegal_B'].append(output_array[b_illegal_indices])
                        # B_sub_activation = sum(output_array[b_legal_indices], output_array[b_omit_indices])
                        # model_analysis_dict['B_token'].append(B_sub_activation / 3)
                        # model_analysis_dict['B_sub'].append(B_sub_activation)
                        # model_analysis_dict['B_cat'].append(B_sub_activation + sum(output_array[b_illegal_indices]))
                        prediction_count_dict[target_category_list[prediction]] += 1
                        token_category_index = self.token_category_index_dict[token_category]
                        if max(output_array[b_legal_indices]) > max(output_array[b_illegal_indices]):
                            correct_count_dict['B_alt_subgroup_correct'] += 1
                        if max(output_array[b_omit_indices]) > max(output_array[b_illegal_indices]):
                            correct_count_dict['B_omit_correct'] += 1
                        target_category_freq_array = self.target_categories.category_freq_array_list[i][j][k]
                        if target_category_list[prediction] == 'B_Legal' \
                                or target_category_list[prediction] == 'B_Present':
                            correct_count_dict['B_subgroup_correct'] += 1
                        if 'B' in target_category_list[prediction]:
                            correct_count_dict['B_group_correct'] += 1
                        if self.model.model_type == 'mlp':
                            if max(output_array[a_legal_indices]) > max(output_array[a_illegal_indices]):
                                correct_count_dict['A_alt_subgroup_correct'] += 1
                            if max(output_array[a_omit_indices]) >= max(output_array[a_illegal_indices]):
                                correct_count_dict['A_omit_correct'] += 1
                        if self.target_categories:
                            for m in range(self.model.vocab_size):
                                output_label = self.model.vocab_list[m]
                                if output_label in self.target_list:
                                    target_category = target_category_list[m]
                                    if target_category == 'B_Present':
                                        target_category = 'B_Legal'
                                    if target_category == 'A_Present':
                                        target_category = 'A_Legal'
                                    target_category_index = self.target_category_index_dict[target_category]
                                    self.output_activation_sum_matrix[
                                        token_category_index, target_category_index] += output_array[m]
                                    target_frequency_array_index = self.target_categories.category_index_dict[
                                        target_category]
                                    target_category_freq = target_category_freq_array[target_frequency_array_index]
                                    if target_category == 'B_Legal' or target_category == 'A_Legal':
                                        target_category_freq += 1
                                    self.output_activation_mean_matrix[
                                        token_category_index, target_category_index] += output_array[
                                                                                            m] / target_category_freq
                    else:
                        if token_category == '.':
                            model_analysis_dict['current_A'].append(output_array[a_current_indices])
                            a_legal_only_indices = list(set(a_legal_indices) - set(a_current_indices))
                            model_analysis_dict['legal_A'].append(output_array[a_legal_only_indices])
                            model_analysis_dict['omit_A'].append(output_array[a_omit_indices])
                            model_analysis_dict['illegal_A'].append(output_array[a_illegal_indices])
                            # A_sub_activation = sum(output_array[a_legal_indices], output_array[a_omit_indices])
                            # model_analysis_dict['A_token'].append(A_sub_activation/3)
                            # model_analysis_dict['A_sub'].append(A_sub_activation)
                            # model_analysis_dict['A_cat'].append(A_sub_activation+sum(output_array[a_illegal_indices]))
                            if target_category_list[prediction] == 'A_Present' \
                                    or target_category_list[prediction] == 'A_Legal':
                                correct_count_dict['A_subgroup_correct'] += 1
                            if 'A' in target_category_list[prediction]:
                                correct_count_dict['A_group_correct'] += 1
                            if max(output_array[a_omit_indices]) >= max(output_array[a_illegal_indices]):
                                correct_count_dict['A_omit_correct'] += 1
                        if token_category == 'A':
                            model_analysis_dict['current_y'].append(sum(output_array[y_indices]) / len(y_indices))
                            if target_category_list[prediction] == 'y':
                                correct_count_dict['y_group_correct'] += 1
                        if token_category == 'B':
                            model_analysis_dict['period'].append(output_array[period_indices])
        model_analysis_dict_final = {}
        for key, value in model_analysis_dict.items():
            model_analysis_dict[key] = np.mean(np.array(model_analysis_dict[key]))
        model_analysis_dict_final['A'] = {'current': model_analysis_dict['current_A'],
                                          'legal': model_analysis_dict['legal_A'],
                                          'omit':model_analysis_dict['omit_A'],
                                          'illegal':model_analysis_dict['illegal_A']}
        model_analysis_dict_final['y'] = {'current': model_analysis_dict['current_y'],
                                          'legal': 0,
                                          'omit': 0,
                                          'illegal': 0}
        model_analysis_dict_final['B'] = {'current': model_analysis_dict['current_B'],
                                          'legal': model_analysis_dict['legal_B'],
                                          'omit' :model_analysis_dict['omit_B'],
                                          'illegal':model_analysis_dict['illegal_B']}
        model_analysis_dict_final['.'] = {'current': model_analysis_dict['period'],
                                          'legal': 0,
                                          'omit': 0,
                                          'illegal': 0}


        self.model_analysis_df = pd.DataFrame(model_analysis_dict_final)
        self.model_analysis_df = self.model_analysis_df.reset_index()
        self.model_analysis_df.rename(columns={'index': 'Type'}, inplace=True)
        self.accuracy_dict = {key: round(value / len(document), 2) for key, value in correct_count_dict.items()}
        sum_matrix_row_sums = self.output_activation_sum_matrix.sum(1)
        safe_inverse = np.divide(1.0, sum_matrix_row_sums, where=sum_matrix_row_sums != 0,
                                 out=np.zeros_like(sum_matrix_row_sums, dtype=float))
        self.output_activation_sum_matrix = self.output_activation_sum_matrix * safe_inverse[:, np.newaxis]
        # print(self.output_activation_mean_matrix)
        mean_matrix_row_sums = self.output_activation_mean_matrix.sum(1)
        safe_inverse = np.divide(1.0, mean_matrix_row_sums, where=mean_matrix_row_sums != 0,
                                 out=np.zeros_like(mean_matrix_row_sums, dtype=float))
        # print(safe_inverse)
        self.output_activation_mean_matrix = self.output_activation_mean_matrix * safe_inverse[:, np.newaxis]
        # print(self.output_activation_mean_matrix)
        # self.print_matrix(self.output_activation_mean_matrix, self.token_category_list, self.target_category_list)
        # self.print_matrix(output_activation_matrix.reshape(-1, output_activation_matrix.shape[2]),
        #                   np.array(self.document_list).flatten().tolist(), self.model.vocab_list)
        self.output_activation_df = pd.DataFrame(np.round(output_activation_matrix.reshape(-1, output_activation_matrix.shape[2]), 2),
                                                 index=np.array(self.document_list).flatten().tolist(),
                                                 columns=self.model.vocab_list)

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
