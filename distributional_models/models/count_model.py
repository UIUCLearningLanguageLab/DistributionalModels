import numpy as np
import copy


class CountModel:
    def __init__(self, corpus, vocab_list, ):
        self.corpus = corpus
        self.vocab_list = vocab_list
        self.input_output_pairs = None
        self.co_occurrence_matrix = None

        self.initialize_matrix()

    def initialize_matrix(self):
        self.co_occurrence_matrix = np.zeros((len(self.vocab_list), len(self.vocab_list)), dtype=int)

    def get_input_output_pairs(self, corpus, sequence):
        x_batches, \
        single_y_batches, \
        y_window_batches = corpus.create_batched_sequence_lists(sequence, 4, 'forward', 1, 1, 'cpu')
        y_batches = single_y_batches

        self.input_output_pairs = [(x_batches[i].view(-1).numpy(), y_batches[i].view(-1).numpy())
                              for i in range(len(x_batches))]

    def count_matrix(self):
        for word1, word2 in self.input_output_pairs:
            idx1 = int(word1)
            idx2 = int(word2)
            self.co_occurrence_matrix[idx1][idx2] += 1
            self.co_occurrence_matrix[idx2][idx1] += 1
