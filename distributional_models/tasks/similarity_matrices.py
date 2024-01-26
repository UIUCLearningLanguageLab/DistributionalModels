import numpy as np


class SimilarityMatrices:

    def __init__(self,
                 embedding_matrix,
                 vocab_index_dict,
                 instance_list=None,
                 instance_categories=None,
                 target_categories=None,
                 instance_target_category_dict=None,
                 svd_dimensions=None,
                 metric='correlation'):

        self.embedding_matrix = embedding_matrix
        self.vocab_index_dict = vocab_index_dict

        self.instance_list = instance_list
        self.instance_index_dict = None
        self.num_instances = None

        self.instance_categories = instance_categories
        self.target_categories = target_categories
        self.instance_target_category_dict = instance_target_category_dict
        self.instance_target_category_list = None
        self.instance_target_category_index_dict = None
        self.num_instance_target_categories = None

        self.svd_dimensions = svd_dimensions
        self.metric = metric

        self.instance_similarity_matrix = None
        self.category_similarity_matrix = None
        self.category_similarity_counts = None

        self.get_instance_list()
        if self.instance_target_category_dict is not None:
            self.get_instance_target_categories()

        self.get_embedding_matrix()
        self.create_similarity_matrix()
        self.create_category_similarity_matrix()

    def get_instance_list(self):
        if self.instance_list is None:
            self.instance_list = list(self.vocab_index_dict.keys())
        else:
            self.num_instances = 0
            for instance in self.instance_list:
                if instance not in self.vocab_index_dict:
                    raise ValueError(f"Instance {instance} not in vocab_index_dict")
                else:
                    self.instance_index_dict[instance] = self.num_instances
                    self.num_instances += 1

    def get_instance_target_categories(self):
        self.instance_target_category_list = []
        self.instance_target_category_index_dict = {}
        self.num_instance_target_categories = 0

        for instance, target_category_dict in self.instance_target_category_dict.items():
            for target, category in target_category_dict.items():
                if category not in self.instance_target_category_list:
                    self.instance_target_category_list.append(category)

        self.num_instance_target_categories = len(self.instance_target_category_list)
        self.instance_target_category_list.sort()
        for i in range(self.num_instance_target_categories):
            self.instance_target_category_index_dict[self.instance_target_category_list[i]] = i

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

    def get_instance_category_matrix_index(self, instance):
        if self.instance_categories is not None:
            category = self.instance_categories.instance_category_dict[instance]
            index = self.instance_categories.instance_category_index_dict[category]
        else:
            index = self.vocab_index_dict[instance]
        return index

    def get_target_category_matrix_index(self, instance, target):
        if self.target_categories is not None:
            category = self.target_categories.instance_category_dict[target]
            index = self.target_categories.instance_category_index_dict[category]
        elif self.instance_target_category_dict is not None:
            index = self.instance_target_category_index_dict[instance][target]
        else:
            index = self.vocab_index_dict[instance]
        return index

    def create_category_similarity_matrix(self):

        if self.instance_categories is not None:
            num_rows = self.instance_categories.num_categories
        else:
            num_rows = len(self.vocab_index_dict)

        if self.target_categories is not None:
            num_columns = self.target_categories.num_categories
        elif self.instance_target_category_dict is not None:
            num_columns = self.num_instance_target_categories
        else:
            num_columns = len(self.vocab_index_dict)

        category_similarity_sums = np.zeros([num_rows, num_columns], float)
        self.category_similarity_counts = np.zeros([num_rows, num_columns], int)

        for i, row_instance in enumerate(self.instance_list):
            row_index = self.get_instance_category_matrix_index(row_instance)
            for j, column_instance in enumerate(self.instance_list):
                column_index = self.get_target_category_matrix_index(column_instance)
                category_similarity_sums[row_index, column_index] += self.instance_similarity_matrix[i, j]
                self.category_similarity_counts[row_index, column_index] += 1

        self.category_similarity_matrix = category_similarity_sums / self.category_similarity_counts

