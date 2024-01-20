import numpy as np
import time
import torch.nn.functional as F


class SequencePredictions:

    def __init__(self, model, sequence_list, sequence_target_label_list=None, token_category_dict=None):
        start_time = time.time()
        self.model = model
        self.sequence_list = sequence_list

        self.token_category_dict = token_category_dict
        self.token_category_list = None
        self.token_category_index_dict = None
        self.num_token_categories = None

        self.sequence_target_label_list = sequence_target_label_list
        self.target_category_list = None
        self.target_category_index_dict = None
        self.num_target_categories = None

        self.token_vocab_index_lists = None
        self.token_category_index_lists = None
        self.target_category_index_lists = None

        self.output_activation_sum_matrix = None

        self.sequences_prediction_accuracy_mean_dict = None
        self.sequence_prediction_accuracy_mean_array = []  # accuracy for each category
        self.sequence_prediction_accuracy_mean = 100.0  # overall accuracy

        self.prepare_data_structures()
        self.get_sequences_prediction_accuracy()
        self.took = time.time() - start_time

    def prepare_data_structures(self):

        self.token_vocab_index_lists = []  # [[2, 8, 13, 1], [2, 9, 13, 1], [], ...]
        self.token_category_index_lists = [] # [[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]]
        self.target_category_index_lists = []  # [[[]], []]

        if self.token_category_dict:
            self.token_category_list = sorted(list(set(self.token_category_dict.values())))
            self.token_category_index_dict = {value: index for index, value in enumerate(self.token_category_list)}
            self.num_token_categories = len(self.token_category_list)

        if self.sequence_target_label_list:
            self.target_category_list = list({item for sublist in self.sequence_target_label_list for item in sublist})
            self.target_category_index_dict = {value: index for index, value in enumerate(self.target_category_list)}
            self.num_target_categories = len(self.target_category_list)

        for i, sequence in enumerate(self.sequence_list):
            token_vocab_index_list = []
            token_category_index_list = []
            target_category_index_list = []
            for j, token in enumerate(sequence):
                token_vocab_index_list.append(self.model.vocab_index_dict[token])
                if self.token_category_dict:
                    token_category_index_list.append(self.token_category_dict[token])
                else:
                    token_category_index_list.append(self.model.vocab_index_dict[token])

                output_category_index_list = []
                for k in range(self.model.vocab_size):
                    if self.sequence_target_label_list:
                        output_category_index = self.target_category_index_dict[self.sequence_target_label_list[i][j][k]]
                    else:
                        output_category_index = k
                    output_category_index_list.append(output_category_index)
                self.target_category_index_list.append(output_category_index_list)

            self.token_vocab_index_lists.append(token_vocab_index_list)
            self.token_category_index_lists.append(token_category_index_list)
            self.target_category_index_lists.append(target_category_index_list)

        vocab_size = len(self.model.vocab_index_dict)

        if self.token_category_dict:
            if self.sequence_target_label_list:
                self.output_activation_sum_matrix = np.zeros([self.num_token_categories, self.num_target_categories], float)
            else:
                self.output_activation_sum_matrix = np.zeros([self.num_token_categories, vocab_size], float)
        else:
            if self.sequence_target_label_list:
                self.output_activation_sum_matrix = np.zeros([vocab_size, self.num_target_categories], float)
            else:
                self.output_activation_sum_matrix = np.zeros([vocab_size, vocab_size], float)

    def get_sequences_prediction_accuracy(self):
        '''
        on each trial
            the activation of actual next predicted item
            the rank of the actual next predicted item

            the average activation of each of the categories
            the summed activation fo each of the categories
            the ranked summed activations of each of the categories

            sequence,
            position,
            token,
            token_category,
            output_activations,
            next_activation,
            next_rank,
            target_category_means,
            target_category_sums,
            target_category_ranks

        :return:
        '''

        self.model.eval()
        for i, sequence in enumerate(self.sequence_list):
            for j, token in enumerate(sequence):
                token_vocab_index = self.token_vocab_index_lists[i][j]
                token_category_index = self.token_category_index_lists[i][j]
                target_category_index_array = self.target_category_index_lists[i][j]
                raw_outputs = self.model([token_vocab_index]).detach()
                probs = F.softmax(raw_outputs).numpy()

                if self.sequence_target_label_list:
                    for k, prob in enumerate(probs):
                        target_category_index = target_category_index_array[k]
                        self.output_activation_sum_matrix[token_category_index, target_category_index] += prob
                else:
                    self.output_activation_sum_matrix[token_category_index, :] += probs

#
# class CategoryPrediction:
#     def __init__(self, model, cohyponym, sequence_list, sequence_list_labels):
#         self.model = model
#         self.cohyponym = cohyponym
#         self.sequence_list = sequence_list
#         self.sequence_list_labels = sequence_list_labels
#         self.took = None
#         self.category_prediction_accuracy_mean_dict = None
#         self.category_prediction_accuracy_mean_array = []  # accuracy for each category
#         self.category_prediction_accuracy_mean = None  # overall accuracy
#
#     def get_category_prediction_accuracy(self):
#         start_time = time.time()
#         self.model.eval()
#         sequences_prediction_accuracy_array = np.empty(len(self.sequence_list_labels),
#                                                        sum(len(sublist) for sublist in self.sequence_list))
#         sequences_prediction_accuracy_dict = defaultdict(list)
#         for i, sequence in enumerate(self.sequence_list):
#             for j, token in enumerate(sequence):
#                 index = self.model.vocab_index_dict[token]
#                 category_labels = np.array(self.sequence_list_labels[i][j])
#                 raw_outputs = self.model([token]).detach()
#                 probs = F.softmax(raw_outputs).numpy()
#                 for k, label_index in enumerate(category_labels):
#                     sequences_prediction_accuracy_dict[label_index].append(probs[k])
#
#         self.sequences_prediction_accuracy_mean_dict = \
#             {key: sum(values) / len(values) if values else 0
#              for key, values in sequences_prediction_accuracy_dict.items()}
#         self.took = time.time() - start_time
#
#


