import copy
import numpy as np


class SequenceCategories:

    def __init__(self, document_list, vocab_index_dict, document_category_lists):
        self.document_list = document_list  # list of documents, which are lists of sequences, which are lists of tokens
        #  [[[a11, y1, b12, .], [a11, y2, b12, .]], [[a11, y1, b12, .], [a11, y2, b12, .]]]
        self.vocab_index_dict = vocab_index_dict
        self.document_category_lists = document_category_lists
        # [[[presentA, y, presentB, .], [a11, y2, b12, .]], [[a11, y1, b12, .], [a11, y2, b12, .]]]
        # [[[[], y, presentB, .], [a11, y2, b12, .]], [[a11, y1, b12, .], [a11, y2, b12, .]]]

        self.vocab_list = None
        self.vocab_size = None

        self.category_list = None
        self.category_index_dict = None
        self.num_categories = None
        self.category_freq_array_list = None

        self.create_vocab_info()
        self.create_category_info()

    def create_vocab_info(self):
        self.vocab_list = list(self.vocab_index_dict.keys())
        self.vocab_size = len(self.vocab_index_dict)

    def create_category_info(self):
        self.category_freq_array_list = copy.deepcopy(self.document_list)
        self.category_list = []
        self.category_index_dict = {}
        self.num_categories = 0

        for i, document in enumerate(self.document_category_lists):
            for j, sequence in enumerate(document):
                for k, token_target_categories in enumerate(sequence):
                    for category in token_target_categories:
                        if category not in self.category_list:
                            self.category_list.append(category)
        self.category_list.sort()

        for category in self.category_list:
            if category not in self.category_index_dict:
                self.category_index_dict[category] = self.num_categories
                self.num_categories += 1

        for i, document in enumerate(self.document_category_lists):
            for j, sequence in enumerate(document):
                for k, token_target_categories in enumerate(sequence):
                    category_freq_array = np.zeros([self.num_categories], int)
                    for category in token_target_categories:
                        category_freq_array[self.category_index_dict[category]] += 1
                    self.category_freq_array_list[i][j][k] = category_freq_array
