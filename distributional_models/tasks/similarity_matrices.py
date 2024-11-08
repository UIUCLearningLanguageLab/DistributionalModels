import copy
import numpy as np
import pandas as pd
from collections import defaultdict

class SimilarityMatrices:

    def __init__(self,
                 embedding_matrix,
                 vocab_index_dict,
                 instance_list=None,
                 instance_categories=None,
                 target_categories=None,
                 instance_target_category_list_dict=None,
                 svd_dimensions=None,
                 metric='correlation'):

        '''
        17x16 embedding_matrix
        17x17 similarity matrix

        5x5 simiarlity

        row dimensions [A1, A2, B1, B2, y] 5
        column_dimensions [A1, A2, B1, B2, y] 5

        row dimensions [A, B, y, .] 4
        column_dimensions [Legal_A, Omitted_A, Present_A, Illegal_A, A,
                            Legal_B, Omitted_B, Present_B, Illegal_B, B,
                            y, .] 12
        '''

        self.embedding_matrix = embedding_matrix
        self.vocab_index_dict = vocab_index_dict
        self.vocab_list = list(self.vocab_index_dict.keys())

        self.instance_list = instance_list
        self.instance_index_dict = None
        self.num_instances = None

        self.instance_categories = instance_categories
        self.instance_category_list = None
        self.instance_category_index_dict = None
        self.num_instance_categories = None

        self.target_categories = target_categories
        self.instance_target_category_list_dict = instance_target_category_list_dict

        self.target_category_list = None
        self.target_category_index_dict = None
        self.num_target_categories = None

        self.svd_dimensions = svd_dimensions
        self.metric = metric

        self.instance_similarity_matrix = None
        self.category_similarity_matrix = None
        self.category_similarity_counts = None

        self.get_instance_list()
        self.get_embedding_matrix()
        self.create_similarity_matrix()

        if self.instance_categories is not None:
            self.get_instance_categories()

        if self.target_categories is not None or self.instance_target_category_list_dict is not None:
            self.get_target_categories()

        if self.instance_category_index_dict is not None or self.target_category_index_dict is not None:
            self.create_category_similarity_matrix()

        if instance_list is not None:
            sub_ranking_dict = self.get_top_similar_words(self.instance_similarity_matrix, self.instance_list, 3)
            model_analysis_dict = defaultdict(list)

            for key, value in sub_ranking_dict.items():
                a_same_sub = 0
                a_oppo_sub = 0
                b_same_sub = 0
                b_oppo_sub = 0
                if self.instance_categories.instance_category_dict[key] == 'A':
                    for word in value:
                        if self.instance_categories.instance_category_dict[word] == 'A':
                            if self.instance_categories.instance_subcategory_dict[key] == self.instance_categories.instance_subcategory_dict[word]:
                                a_same_sub += 1
                            else:
                                a_oppo_sub += 1
                    model_analysis_dict['a_same_sub'].append(a_same_sub / 3)
                    model_analysis_dict['a_oppo_sub'].append(a_oppo_sub / 3)
                else:
                    for word in value:
                        if self.instance_categories.instance_category_dict[word] == 'B':
                            if self.instance_categories.instance_subcategory_dict[key] == self.instance_categories.instance_subcategory_dict[word]:
                                b_same_sub += 1
                            else:
                                b_oppo_sub += 1
                    model_analysis_dict['b_same_sub'].append(b_same_sub / 3)
                    model_analysis_dict['b_oppo_sub'].append(b_oppo_sub / 3)

            cat_ranking_dict = self.get_top_similar_words(self.instance_similarity_matrix, instance_list, 6)

            for key, value in cat_ranking_dict.items():
                a_oppo_cat = 0
                b_oppo_cat = 0
                if self.instance_categories.instance_category_dict[key] == 'A':
                    for word in value:
                        if self.instance_categories.instance_category_dict[word] != 'A':
                            a_oppo_cat += 1
                    model_analysis_dict['a_oppo_cat'].append(a_oppo_cat / 6)
                else:
                    for word in value:
                        if self.instance_categories.instance_category_dict[word] != 'B':
                            b_oppo_cat += 1
                    model_analysis_dict['b_oppo_cat'].append(a_same_sub / 6)
            model_analysis_dict_final = {}
            for key, value in model_analysis_dict.items():
                model_analysis_dict[key] = np.round(np.mean(np.array(model_analysis_dict[key])), 2)
            model_analysis_dict_final['A'] = {'subcategory': model_analysis_dict['a_same_sub'],
                                              'opposite subcategory': model_analysis_dict['a_oppo_sub'],
                                              'opposite category': model_analysis_dict['a_oppo_cat']}
            model_analysis_dict_final['B'] = {'subcategory': model_analysis_dict['b_same_sub'],
                                              'opposite subcategory': model_analysis_dict['b_oppo_sub'],
                                              'opposite category': model_analysis_dict['b_oppo_cat']}

            self.model_analysis_df = pd.DataFrame(model_analysis_dict_final)
            self.model_analysis_df = self.model_analysis_df.reset_index()
            self.model_analysis_df.rename(columns={'index': 'Type'}, inplace=True)

    def get_instance_list(self):
        # the list of words that are used to build the similarity matrices
        # if None, will use the whole vocab list
        if self.instance_list is None:
            self.instance_list = list(self.vocab_index_dict.keys())
            self.num_instances = len(self.instance_list)
            self.instance_index_dict = copy.deepcopy(self.vocab_index_dict)
        else:
            self.num_instances = 0
            self.instance_index_dict = {}
            for instance in self.instance_list:
                if instance not in self.vocab_index_dict:
                    raise ValueError(f"Instance {instance} not in vocab_index_dict")
                else:
                    self.instance_index_dict[instance] = self.num_instances
                    self.num_instances += 1

    def get_embedding_matrix(self):
        reduced_embedding_matrix = np.zeros([self.num_instances, self.embedding_matrix.shape[1]])
        for i, instance in enumerate(self.instance_list):
            index = self.vocab_index_dict[instance]
            reduced_embedding_matrix[i, :] = self.embedding_matrix[index, :]

        if self.svd_dimensions is not None:
            u, s, v = np.linalg.svd(reduced_embedding_matrix)
            self.embedding_matrix = u[:, :self.svd_dimensions]
        else:
            self.embedding_matrix = reduced_embedding_matrix

    def create_similarity_matrix(self):
        if self.metric == "correlation":
            self.instance_similarity_matrix = np.corrcoef(self.embedding_matrix)
        elif self.metric == "cosine":
            norm_matrix = self.embedding_matrix / np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            self.instance_similarity_matrix = np.dot(norm_matrix, norm_matrix.T)
        else:
            raise ValueError(f"Unrecognized similarity metric {self.metric}")

    def get_instance_categories(self):
        self.instance_category_list = []
        self.instance_category_index_dict = {}
        self.num_instance_categories = 0

        for instance in self.instance_list:
            if instance in self.instance_categories.instance_category_dict:
                category = self.instance_categories.instance_category_dict[instance]
                if category not in self.instance_category_list:
                    self.instance_category_list.append(category)
                    self.instance_category_index_dict[category] = self.num_instance_categories
                    self.num_instance_categories += 1
            else:
                raise Exception("Instance not in instance_categories")

    def get_target_categories(self):
        self.target_category_list = []
        self.target_category_index_dict = {}
        self.num_target_categories = 0

        if self.target_categories is not None:
            for target in self.instance_list:
                if target in self.target_categories.instance_category_dict:
                    category = self.target_categories.instance_category_dict[target]
                    if category not in self.instance_category_list:
                        self.target_category_list.append(category)
                        self.target_category_index_dict[category] = self.num_target_categories
                        self.num_target_categories += 1
                else:
                    raise Exception("Target not in target_categories")

        elif self.instance_target_category_list_dict is not None:
            # {'A1_1': ['OTHER', '.', 'Present_A']}

            for target in self.instance_list:
                if target in self.instance_target_category_list_dict:
                    target_category_list = self.instance_target_category_list_dict[target]
                    for i, target_category in enumerate(target_category_list):
                        vocab_word = self.vocab_list[i]
                        if vocab_word in self.instance_index_dict:
                            if target_category not in self.target_category_list:
                                self.target_category_list.append(target_category)
                    self.target_category_list.sort()
                    self.num_target_categories = len(self.target_category_list)
                    for i in range(self.num_target_categories):
                        self.target_category_index_dict[self.target_category_list[i]] = i
                else:
                    raise Exception("Target not in instance_target_category_list_dict")
        else:
            raise Exception("Must provide either target_categories or instance_target_category_list_dict")

    def get_instance_category_matrix_index(self, instance):
        if self.instance_categories is not None:
            category = self.instance_categories.instance_category_dict[instance]
            category_index = self.instance_category_index_dict[category]
        else:
            category = None
            category_index = self.vocab_index_dict[instance]
        return category_index

    def get_target_category_matrix_index(self, instance, target):
        if self.target_categories is not None:
            category = self.target_categories.instance_category_dict[target]
            category_index = self.instance_category_index_dict[category]
        elif self.instance_target_category_list_dict is not None:
            target_index = self.vocab_index_dict[target]
            category = self.instance_target_category_list_dict[instance][target_index]
            category_index = self.target_category_index_dict[category]
        else:
            category = None
            category_index = self.vocab_index_dict[target]
        return category_index

    def create_category_similarity_matrix(self):
        if self.instance_categories is not None:
            # num_rows = self.instance_categories.num_categories
            num_rows = self.num_instance_categories
        else:
            num_rows = len(self.vocab_index_dict)

        if self.target_categories is not None:
            num_columns = self.num_target_categories
        elif self.instance_target_category_list_dict is not None:
            num_columns = self.num_target_categories
        else:
            num_columns = len(self.vocab_index_dict)

        category_similarity_sums = np.zeros([num_rows, num_columns], float)
        self.category_similarity_counts = np.zeros([num_rows, num_columns], int)

        for i, row_instance in enumerate(self.instance_list):
            row_index = self.get_instance_category_matrix_index(row_instance)
            for j, column_instance in enumerate(self.instance_list):
                column_index = self.get_target_category_matrix_index(row_instance, column_instance)
                sim_score = self.instance_similarity_matrix[i, j]
                category_similarity_sums[row_index, column_index] += sim_score
                self.category_similarity_counts[row_index, column_index] += 1
        # self.print_matrix(category_similarity_sums, self.instance_category_list, self.target_category_list)
        # self.print_matrix(self.category_similarity_counts, self.instance_category_list, self.target_category_list)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.category_similarity_matrix = category_similarity_sums / self.category_similarity_counts
        # self.print_matrix(self.category_similarity_matrix, self.instance_category_list, self.target_category_list)

    def output_results(self):
        print()
        print("instance_similarity_matrix")
        self.print_matrix(self.instance_similarity_matrix, self.instance_list, self.instance_list)
        print()
        print("category_similarity_matrix")
        self.print_matrix(self.category_similarity_matrix, self.instance_category_list, self.target_category_list)
        print()

    @staticmethod
    def print_matrix(matrix, row_labels, column_labels):
        # Determine the width of each column
        col_width = 10

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

    @staticmethod
    def get_top_similar_words(similarity_matrix, vocab_list, top_n=3):
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Similarity matrix must be square.")
        if len(vocab_list) != similarity_matrix.shape[0]:
            raise ValueError("Length of vocab_list must match the dimensions of the similarity matrix.")

        top_similar_words = {}
        for i, word in enumerate(vocab_list):
            # Get the similarity scores for the current word
            similarity_scores = similarity_matrix[i]
            # # Get the indices of the top N most similar words, excluding the word itself
            # top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
            # Get the indices of the top N most similar words, not excluding the word itself
            top_indices = np.argsort(similarity_scores)[::-1][:top_n]
            # Get the corresponding words from vocab_list
            top_words = [vocab_list[idx] for idx in top_indices]
            # Store the results in the dictionary
            top_similar_words[word] = top_words

        return top_similar_words