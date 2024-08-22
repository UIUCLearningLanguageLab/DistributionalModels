import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from scipy.stats import pearsonr
from collections import defaultdict, OrderedDict


class States:
    def __init__(self, model, corpus, params, save_path=None, layer='hidden'):
        self.model = model
        self.corpus = corpus
        self.params = params
        self.save_path = save_path
        self.layer = layer
        self.input_sequences = None
        self.contrast_pairs = None
        self.input_embed = None
        self.pre_y1_states = None
        self.y1_output_activations = None

        self.input_weights = None
        self.hidden_weights = None
        self.output_weights = None
        self.hidden_state = None
        self.input_bias_weights = None
        self.hidden_bias_weights = None
        self.output_bias_weights = None
        self.hidden_size = None

        self.token_hidden_state_dict = None
        self.token_category_hidden_state_dict = None
        self.token_subcategory_hidden_state_dict = None
        self.token_contrast_distribution_dict = None
        self.token_pre_hidden_state_dict = None
        self.sequence_hidden_state_dict = None
        self.sequence_pre_hidden_state_dict = None
        self.average_token_hidden_state_dict = None
        self.average_token_pre_hidden_state_dict = None
        self.vocab_category_dict = None
        self.vocab_subcategory_dict = None
        # self.get_similarity_between_tokens(self.average_token_hidden_state_dict)

    def generate_input_sequences(self):
        self.input_sequences = [['y3', 'B2_1', '.', 'A1_1', 'y2'],
                                ['y3', 'B2_1', '.', 'A1_2', 'y2'],
                                ['y3', 'B2_1', '.', 'A1_3', 'y2'],
                                ['y3', 'B2_1', '.', 'A2_1', 'y1'],
                                ['y3', 'B2_1', '.', 'A2_2', 'y1'],
                                ['y3', 'B2_1', '.', 'A2_3', 'y1']]

    def generate_contrast_pairs(self):
        self.contrast_pairs = [[['.'], ['y']],
                               [['.'], ['B']],
                               [['y'], ['B']],
                               [['A'], ['B']],
                               [['B1'], ['B2']],
                               [['B1_1'], ['B1_2']],
                               [['B1_2'], ['B1_3']],
                               [['B1_1'], ['B1_3']],
                               [['y'], ['A']],
                               [['A1'], ['A2']],
                               [['A1_1'], ['A1_2']],
                               [['A1_2'], ['A1_3']],
                               [['A1_1'], ['A1_3']]]

    def get_vocab_category_subcategory_into(self):
        self.vocab_category_dict = self.corpus.create_word_category_dict(self.model.vocab_index_dict)
        self.vocab_subcategory_dict = self.corpus.create_word_category_dict(self.model.vocab_index_dict, True)

    @staticmethod
    def compute_entropy_of_hidden_state(hidden_state, add_one=False):
        add_one_array = np.array([[1]])
        if hidden_state.ndim == 1:
            hidden_state = np.expand_dims(hidden_state, axis=0)
        if add_one:
            hidden_state = np.concatenate((hidden_state, add_one_array), axis=1)
        min_value = np.min(hidden_state)
        shifted_hidden_state = hidden_state - min_value + 1
        probability_distribution = shifted_hidden_state / shifted_hidden_state.sum()
        flatten_hidden = probability_distribution.flatten()
        entropy = stats.entropy(flatten_hidden)
        return entropy

    @staticmethod
    def calculate_sparsity(hidden_state, threshold=0.1):
        sparsity = np.mean(np.abs(hidden_state) < threshold)
        return sparsity

    def get_hidden_state(self, sequence):
        model = self.model
        model.eval()
        _, _, hidden_states = self.model.test_sequence(self.corpus, sequence, self.params)
        return hidden_states[-1]

    def get_hidden_states(self, compute_entropy=False, compute_sparsity=False, compute_contrast_contribution=False):
        model = self.model
        if compute_entropy or compute_sparsity:
            vocab_category_info_dict = self.corpus.create_word_category_dict(model.vocab_index_dict)
            vocab_subcategory_info_dict = self.corpus.create_word_category_dict(model.vocab_index_dict, True)
        model.eval()
        test_document_list = copy.deepcopy(self.corpus.document_list)
        self.token_hidden_state_dict = defaultdict(list)
        self.token_pre_hidden_state_dict = defaultdict(list)
        self.token_category_hidden_state_dict = defaultdict(list)
        self.token_subcategory_hidden_state_dict = defaultdict(list)
        self.token_contrast_distribution_dict = {}
        token_contrast_distribution_dict = {}
        self.sequence_hidden_state_dict = defaultdict(list)
        self.sequence_pre_hidden_state_dict = defaultdict(list)
        for document in test_document_list:
            for sequence in document:
                pre_hidden_states, _, hidden_states = model.test_sequence(self.corpus, sequence, self.params)
                for i, token in enumerate(sequence):
                    self.token_hidden_state_dict[token].append(hidden_states[i])
                    if compute_entropy:
                        if vocab_category_info_dict[token] not in ['.', 'unknown']:
                            hidden_entropy = States.compute_entropy_of_hidden_state(hidden_states[i], True)
                            self.token_category_hidden_state_dict[vocab_subcategory_info_dict[token]].append(
                                hidden_entropy)
                            if vocab_category_info_dict[token] != 'y':
                                self.token_category_hidden_state_dict[vocab_category_info_dict[token]].append(
                                    hidden_entropy)
                    elif compute_sparsity:
                        if vocab_category_info_dict[token] not in ['.', 'unknown']:
                            hidden_sparsity = States.calculate_sparsity(hidden_states[i])
                            self.token_category_hidden_state_dict[vocab_subcategory_info_dict[token]].append(
                                hidden_sparsity)
                            if vocab_category_info_dict[token] != 'y':
                                self.token_category_hidden_state_dict[vocab_category_info_dict[token]].append(
                                    hidden_sparsity)
                    elif compute_contrast_contribution:
                        contrast_contribution_dict = {}
                        for pair in self.contrast_pairs:
                            key = ''.join(pair[0]) + 'vs.' + ''.join(pair[1])
                            contrast_contributions = self.compute_contrast_contribution(
                                    hidden_states[i], pair[0], pair[1]).tolist()
                            for hidden_index, contrast_contribution in enumerate(contrast_contributions):
                                contrast_contribution_dict[key+str(hidden_index)] = np.round(contrast_contribution,2)
                        self.token_contrast_distribution_dict[token] = contrast_contribution_dict
                        self.token_contrast_distribution_dict = OrderedDict(sorted(
                            self.token_contrast_distribution_dict.items()))

                    if model.model_type == 'lstm' or model.model_type == 'srn':
                        self.token_pre_hidden_state_dict[token].append(pre_hidden_states[i])
                self.sequence_hidden_state_dict[' '.join(sequence)].append(hidden_states)
                if model.model_type == 'lstm' or model.model_type == 'srn':
                    self.sequence_pre_hidden_state_dict[' '.join(sequence)].append(pre_hidden_states)
        self.average_token_hidden_state_dict = {key: np.mean(np.stack(value), axis=0)
                                                for key, value in self.token_hidden_state_dict.items()}
        if model.model_type == 'lstm' or model.model_type == 'srn':
            self.average_token_pre_hidden_state_dict = {key: np.mean(np.stack(value), axis=0)
                                                        for key, value in self.token_pre_hidden_state_dict.items()}
        token_category_subcategory_hidden_state_dict = {**self.token_category_hidden_state_dict,
                                                        **self.token_subcategory_hidden_state_dict}
        if compute_entropy or compute_sparsity:
            return {key: np.round(np.mean(value), 2) for key, value in token_category_subcategory_hidden_state_dict.items()}

    def get_weights(self):
        self.input_weights = self.model.get_weights('input')
        self.input_bias_weights = self.model.get_weights('input_bias')
        self.output_weights = self.model.get_weights('output')
        self.hidden_size = self.output_weights.shape[1]
        self.output_bias_weights = self.model.get_weights('output_bias')
        if self.model.model_type != 'mlp':
            self.hidden_weights = self.model.get_weights('hidden')
            self.hidden_bias_weights = self.model.get_weights('hidden_bias')
            if self.model.model_type == 'lstm':
                self.input_weights = self.input_weights[:, -self.hidden_size:]
                self.input_bias_weights = self.input_bias_weights[-self.hidden_size:]
                self.hidden_weights = self.hidden_weights[:, -self.hidden_size:]
                self.hidden_bias_weights = self.hidden_bias_weights[-self.hidden_size:]

    def export_weights_to_csv(self):

        combined_input_weights = np.concatenate((np.expand_dims(self.input_bias_weights, axis=0),
                                                 self.input_weights), axis=0)
        combined_output_weights = np.concatenate((np.expand_dims(self.output_bias_weights, axis=1),
                                                 self.output_weights), axis=1)


    def compute_r_score(self, category1, category2, type='global'):
        r_scores = []
        contrast_vector = np.zeros([len(self.model.vocab_list)])
        if set(category1).issubset(self.vocab_category_dict.values()) and set(category2).issubset(
                self.vocab_category_dict.values()):
            category1_indices = [i for i, (key, value) in enumerate(self.vocab_category_dict.items())
                                 if value in category1]
            category2_indices = [i for i, (key, value) in enumerate(self.vocab_category_dict.items())
                                 if value in category2]
        elif set(category1).issubset(self.vocab_subcategory_dict.values()) and set(category2).issubset(
                self.vocab_subcategory_dict.values()):
            category1_indices = [i for i, (key, value) in enumerate(self.vocab_subcategory_dict.items())
                                 if value in category1]
            category2_indices = [i for i, (key, value) in enumerate(self.vocab_subcategory_dict.items())
                                 if value in category2]
        else:
            category1_indices = [i for i, value in enumerate(self.model.vocab_list)
                                 if value in category1]
            category2_indices = [i for i, value in enumerate(self.model.vocab_list)
                                 if value in category2]
        contrast_vector[category1_indices] = 1
        contrast_vector[category2_indices] = -1
        output_bias_reshaped = self.output_bias_weights.reshape(-1, 1)
        output_and_bias_weights = np.concatenate((output_bias_reshaped, self.output_weights), axis=1)
        if type == 'specific':
            contrast_vector = contrast_vector[contrast_vector != 0]
            output_and_bias_weights = output_and_bias_weights[category1_indices+category2_indices, :]

        for i in range(output_and_bias_weights.shape[1]):
            column = output_and_bias_weights[:, i]
            correlation, _ = pearsonr(contrast_vector, column)
            r_scores.append(correlation)
        return np.round(np.array(r_scores),2)

    def compute_contrast_contribution(self, hidden_state, category1, category2):
        if set(category1).issubset(self.vocab_category_dict.values()) and set(category2).issubset(
                self.vocab_category_dict.values()):
            category1_indices = [i for i, (key, value) in enumerate(self.vocab_category_dict.items())
                                 if value in category1]
            category2_indices = [i for i, (key, value) in enumerate(self.vocab_category_dict.items())
                                 if value in category2]
        elif set(category1).issubset(self.vocab_subcategory_dict.values()) and set(category2).issubset(
                self.vocab_subcategory_dict.values()):
            category1_indices = [i for i, (key, value) in enumerate(self.vocab_subcategory_dict.items())
                                 if value in category1]
            category2_indices = [i for i, (key, value) in enumerate(self.vocab_subcategory_dict.items())
                                 if value in category2]
        else:
            category1_indices = [i for i, value in enumerate(self.model.vocab_list)
                                 if value in category1]
            category2_indices = [i for i, value in enumerate(self.model.vocab_list)
                                 if value in category2]

        hidden_state = np.concatenate((hidden_state.reshape(1, -1), np.array([[1]])), axis=1)
        output_bias_reshaped = self.output_bias_weights.reshape(-1, 1)
        output_and_bias_weights = np.concatenate((output_bias_reshaped, self.output_weights), axis=1)
        category1_weights = output_and_bias_weights[category1_indices, :]
        category2_weights = output_and_bias_weights[category2_indices, :]
        category1_contribution = np.mean((hidden_state * category1_weights), axis=0)
        category2_contribution = np.mean((hidden_state * category2_weights), axis=0)
        contribution_difference = np.abs(category1_contribution - category2_contribution)
        return contribution_difference/ np.sum(contribution_difference)



    def get_similarity_between_tokens(self, rep_dict, metric='correlation'):
        token_list = copy.deepcopy(self.model.vocab_list)[1:]
        similarity_matrix = np.full((len(token_list), len(token_list)), None)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                token1 = token_list[i]
                token2 = token_list[j]
                rep_state1 = rep_dict[token1]
                rep_state2 = rep_dict[token2]
                if np.all(rep_state1 == 0) or np.all(rep_state2 == 0):
                    # Skip the calculation and set the correlation to None or some indicator value
                    correlation = None
                else:
                    if metric == 'correlation':
                        correlation, _ = pearsonr(rep_state1, rep_state2)

                similarity_matrix[i, j] = correlation

        similarity_df = pd.DataFrame(similarity_matrix, index=token_list, columns=token_list)
        print('getting correlation')

    def evaluate_states(self, position=-1):
        if self.model is not None:
            self.model.eval()
            state_dict_list = []
            fig_dict = {}
            for sequence in self.input_sequences:
                if self.model.model_type == 'transformer':
                    input_states = [self.model.layer_dict['token_embeddings_table']
                                    (torch.tensor([[self.model.vocab_index_dict[token]]])).squeeze(1).detach().numpy()
                                    for token in sequence]
                else:
                    input_states = [self.model.layer_dict['embedding']
                                    (torch.tensor([[self.model.vocab_index_dict[token]]])).squeeze(1).numpy()
                                    for token in sequence]
                # input_weights = [self.input_weights[self.model.vocab_index_dict[token]].reshape(1, self.hidden_size)
                #                  for token in sequence]
                if self.model.model_type == 'transformer':
                    [word_batch_list, combined_inputs, weighted_attention_values], output_activations, hidden_states = \
                        self.model.test_sequence(self.corpus, sequence, self.params, softmax=False)
                else:
                    pre_hidden_states, output_activations, hidden_states = \
                        self.model.test_sequence(self.corpus, sequence, self.params, softmax=False)
                if self.hidden_weights is None or self.model.model_type == 'transformer':
                    pre_hidden_states_in_pos = None
                    reshaped_hidden_bias_weights = None
                else:
                    pre_hidden_states_in_pos = pre_hidden_states[position].reshape(1, self.hidden_size)
                    reshaped_hidden_bias_weights = self.hidden_bias_weights.reshape(self.hidden_size, 1)
                if self.model.model_type == 'transformer':
                    fig = self.visualize_transformer_learning(word_batch_list[position], combined_inputs[position],
                                                              weighted_attention_values[position],
                                                              self.hidden_weights.transpose(1, 0),
                                                              self.hidden_bias_weights.reshape(self.hidden_size, 1),
                                                              hidden_states[position].reshape(1, self.hidden_size),
                                                              self.output_weights,
                                                              self.output_bias_weights.reshape(len(self.model.vocab_list), 1),
                                                              output_activations[position].reshape(1, len(self.model.vocab_list)))
                else:
                    fig = self.visualize_learning(input_states[position], self.input_weights.transpose(1, 0),
                                                  self.input_bias_weights.reshape(self.hidden_size, 1),
                                                  hidden_states[position].reshape(1, self.hidden_size),
                                                  self.output_weights,
                                                  self.output_bias_weights.reshape(len(self.model.vocab_list), 1),
                                                  output_activations[position].reshape(1, len(self.model.vocab_list)),
                                                  pre_hidden_states_in_pos,
                                                  self.hidden_weights,
                                                  reshaped_hidden_bias_weights,
                                                  )
                fig_dict[sequence[-2]] = fig

            return fig_dict

    def softmax(self, z_output):
        exp_x = np.exp(z_output - np.max(z_output, axis=1, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x

    def visualize_transformer_learning(self, word_batch, combined_input_states, weighted_attention_values,
                                       weighted_attention_weights, weighted_attention_bias_weights,
                                       hidden_states, output_weights, output_bias_weights, output_activation):

        if self.model.embedding_size == 4:
            fig_width = 9
            fig_height = 12
            cell_width = 0.45 / 17
            input_right_pos = 0.45
        else:
            fig_width = 15
            fig_height = 15
            cell_width = 0.33 / 17
            input_right_pos = 0.35
        offset = 0.015
        fig = plt.figure(figsize=(fig_width, fig_height))
        pos1 = [input_right_pos, 0.025, cell_width * combined_input_states.shape[1], cell_width * combined_input_states.shape[0]]
        pos2 = [input_right_pos, offset + pos1[1] + pos1[3], cell_width * weighted_attention_values.shape[1],
                cell_width * weighted_attention_values.shape[0]]
        pos3 = [input_right_pos, offset + pos2[1] + pos2[3], cell_width * weighted_attention_weights.shape[1],
                cell_width * weighted_attention_weights.shape[0]]
        pos4 = [pos3[0] - 0.1, pos3[1], cell_width * weighted_attention_bias_weights.shape[1],
                cell_width * weighted_attention_bias_weights.shape[0]]
        pos5 = [input_right_pos, offset + pos3[1] + pos3[3], cell_width * hidden_states.shape[1],
                cell_width * hidden_states.shape[0]]
        pos6 = [pos5[0], pos5[1] + pos5[3] + offset, cell_width * output_weights.shape[1],
                cell_width * output_weights.shape[0]]
        pos7 = [pos6[0] - 0.1, pos6[1], cell_width * output_bias_weights.shape[1],
                cell_width * output_bias_weights.shape[0]]
        pos8 = [0.33, pos7[1] + pos7[3] + offset, cell_width * output_activation.shape[1],
                 cell_width * output_activation.shape[0]]
        pos9 = [0.33, pos8[1] + pos8[3] + offset, cell_width * output_activation.shape[1],
                 cell_width * output_activation.shape[0]]

        ax1 = fig.add_axes(pos1)
        ax2 = fig.add_axes(pos2)
        ax3 = fig.add_axes(pos3)
        ax4 = fig.add_axes(pos4)
        ax5 = fig.add_axes(pos5)
        ax6 = fig.add_axes(pos6)
        ax7 = fig.add_axes(pos7)
        ax8 = fig.add_axes(pos8)
        ax9 = fig.add_axes(pos9)

        ax1 = sns.heatmap(combined_input_states, ax=ax1, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=word_batch, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax1.set(xlabel="input state")
        ax2 = sns.heatmap(weighted_attention_values, ax=ax2, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=word_batch, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax2.set(xlabel="attention weighted values")
        ax3 = sns.heatmap(weighted_attention_weights, ax=ax3, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=False, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax3.set(xlabel="input to hidden weights")
        ax4 = sns.heatmap(weighted_attention_bias_weights, ax=ax4, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=False, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax4.set(xlabel="bias")
        ax5 = sns.heatmap(hidden_states, ax=ax5, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=False, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax5.set(xlabel="hidden state")
        ax6 = sns.heatmap(output_weights, ax=ax6, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=self.model.vocab_list, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax6.set(xlabel="output weights")
        ax7 = sns.heatmap(output_bias_weights, ax=ax7, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                          annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax7.set(ylabel="output bias weights")
        ax8 = sns.heatmap(output_activation, ax=ax8, cmap='flare', cbar=False, xticklabels=False,
                          yticklabels=False,
                          annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax9 = sns.heatmap(self.softmax(output_activation), ax=ax9, cmap='flare', cbar=False, xticklabels=self.model.vocab_list,
                           yticklabels=False,
                           annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax9.xaxis.tick_top()
        ax9.set(xlabel="output activation")
        ax9.set_xticklabels(ax9.get_xticklabels(), rotation=90, ha='left')

        return fig

    def visualize_learning(self, input_states, input_weights, input_bias_weights, hidden_states,
                           output_weights, output_bias_weights, output_activation, pre_hidden_states=None,
                           hidden_weights=None, hidden_bias_weights=None):

        if self.hidden_size == 16:
            # Define positions and sizes for each heatmap [left, bottom, width, height]
            fig_width = 15
            fig_height = 15
            cell_width = 0.35 / 17
            output_weights_right_pos = 0.35
        else:
            fig_width = 8.5
            fig_height = 10
            cell_width = 0.5 / 17
            output_weights_right_pos = 0.45
        offset = 0.02
        fig = plt.figure(figsize=(fig_width, fig_height))
        pos1 = [0.2, 0.1, cell_width * input_states.shape[1], cell_width * input_states.shape[0]]
        pos2 = [pos1[0], pos1[1] + pos1[3] + offset, cell_width * input_weights.shape[1],
                cell_width * input_weights.shape[0]]
        pos3 = [pos2[0] - 0.1, pos2[1], cell_width * input_bias_weights.shape[1],
                cell_width * input_bias_weights.shape[0]]
        pos4 = [output_weights_right_pos, pos2[1] + pos2[3] + offset, cell_width * hidden_states.shape[1],
                cell_width * hidden_states.shape[0]]
        if pre_hidden_states is not None:
            pos5 = [pos1[0] + pos1[2] + offset, pos1[1], cell_width * pre_hidden_states.shape[1],
                    cell_width * pre_hidden_states.shape[0]]
            pos6 = [pos5[0], pos2[1], cell_width * hidden_weights.shape[1], cell_width * hidden_weights.shape[0]]
            pos7 = [pos6[0] + pos6[2] + 0.05, pos6[1], cell_width * hidden_bias_weights.shape[1],
                    cell_width * hidden_bias_weights.shape[0]]
        pos8 = [pos4[0], pos4[1] + pos4[3] + offset, cell_width * output_weights.shape[1],
                cell_width * output_weights.shape[0]]
        pos9 = [pos8[0] - 0.15, pos8[1], cell_width * output_bias_weights.shape[1],
                cell_width * output_bias_weights.shape[0]]
        pos10 = [0.2, pos8[1] + pos8[3] + offset, cell_width * input_states.shape[1],
                 cell_width * input_states.shape[0]]
        pos11 = [0.2, pos10[1] + pos10[3] + offset, cell_width * input_states.shape[1],
                 cell_width * input_states.shape[0]]

        # Create axes using the specified positions
        ax1 = fig.add_axes(pos1)
        ax2 = fig.add_axes(pos2)
        ax3 = fig.add_axes(pos3)
        ax4 = fig.add_axes(pos4)
        if pre_hidden_states is not None:
            ax5 = fig.add_axes(pos5)
            ax6 = fig.add_axes(pos6)
            ax7 = fig.add_axes(pos7)
        ax8 = fig.add_axes(pos8)
        ax9 = fig.add_axes(pos9)
        ax10 = fig.add_axes(pos10)
        ax11 = fig.add_axes(pos11)

        # Plot heatmaps
        ax1 = sns.heatmap(input_states, ax=ax1, cmap='flare', cbar=False, xticklabels=self.model.vocab_list,
                          yticklabels=False, annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax1.set(xlabel="input state")
        ax2 = sns.heatmap(input_weights, ax=ax2, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                          annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax2.set(xlabel="input weight")
        ax3 = sns.heatmap(input_bias_weights, ax=ax3, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                          annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax3.set(ylabel="input bias weight")
        ax4 = sns.heatmap(hidden_states, ax=ax4, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                          annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax4.set(xlabel="hidden state")
        if pre_hidden_states is not None:
            ax5 = sns.heatmap(pre_hidden_states, ax=ax5, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                              annot=True,
                              fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
            ax5.yaxis.tick_right()
            ax5.set(xlabel="pre hidden state")
            ax6 = sns.heatmap(hidden_weights, ax=ax6, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                              annot=True,
                              fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
            ax6.yaxis.tick_right()
            ax6.set(xlabel="hidden weight")
            ax7 = sns.heatmap(hidden_bias_weights, ax=ax7, cmap='flare', cbar=False, xticklabels=False,
                              yticklabels=False, annot=True,
                              fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
            ax7.yaxis.tick_right()
            ax7.set(ylabel="hidden bias weight")
        ax8 = sns.heatmap(output_weights, ax=ax8, cmap='flare', cbar=False, xticklabels=False, annot=True,
                          fmt=".2f", annot_kws={"size": 6},
                          yticklabels=self.model.vocab_list, square=True, linewidths=0.5, linecolor='white')
        ax8.set(ylabel="output weights")
        ax9 = sns.heatmap(output_bias_weights, ax=ax9, cmap='flare', cbar=False, xticklabels=False, yticklabels=False,
                          annot=True,
                          fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax9.set(ylabel="output bias weights")
        ax10 = sns.heatmap(output_activation, ax=ax10, cmap='flare', cbar=False, xticklabels=False,
                           yticklabels=False,
                           annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax11 = sns.heatmap(self.softmax(output_activation), ax=ax11, cmap='flare', cbar=False, xticklabels=self.model.vocab_list,
                           yticklabels=False,
                           annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.5, linecolor='white')
        ax11.xaxis.tick_top()
        ax11.set(xlabel="output activation")
        ax11.set_xticklabels(ax11.get_xticklabels(), rotation=90, ha='left')

        return fig
