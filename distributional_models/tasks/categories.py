import random
import csv
import numpy as np
from torch.utils.data import Dataset, random_split


class Categories(Dataset):

    def __init__(self, instance_category_dict=None, instance_category_file_path=None):

        self.category_list = None
        self.category_index_dict = None
        self.instance_category_dict = instance_category_dict
        self.instance_category_file_path = instance_category_file_path

        self.num_categories = None
        self.instance_category_index_dict = None

        self.instance_list = None
        self.instance_index_dict = None
        self.num_instances = None
        self.instance_size = None  # the size of the feature vector, embedding vector

        self.category_instance_list_dict = None
        self.instance_category_matrix = None
        self.instance_instance_matrix = None

        self.sequence_target_category_list = None
        self.target_category_list = None
        self.target_category_index_dict = None
        self.num_target_categories = None

        self.instance_feature_matrix = None
        self.x_index_list = None
        self.x_list = []
        self.y_list = []

        if self.instance_category_dict is not None:
            self.create_from_instance_category_dict()
        elif self.instance_category_file_path is not None:
            self.create_from_category_file()
        else:
            raise Exception("Must provide either an instance_category_dict or a instance_category_file_path")

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.instance_list, self.x_list[idx], self.y_list[idx]

    def create_from_category_file(self):

        self.instance_category_dict = {}

        with open(self.instance_category_file_path, encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            for row in reader:
                key = row[0]
                if key in self.instance_category_dict:
                    raise ValueError(f"Duplicate instance detected: {key}")
                self.instance_category_dict[key] = row[1]

        self.create_from_instance_category_dict()

    def create_from_instance_category_dict(self):
        self.category_index_dict = {}
        self.category_list = []
        self.num_categories = 0
        for instance in self.instance_category_dict:
            category = self.instance_category_dict[instance]
            if category not in self.category_index_dict:
                self.category_index_dict[category] = self.num_categories
                self.num_categories += 1
                self.category_list.append(category)

        self.category_instance_list_dict = {}

        for i in range(self.num_categories):
            self.category_instance_list_dict[self.category_list[i]] = []

        self.instance_list = list(self.instance_category_dict.keys())
        self.num_instances = len(self.instance_list)

        self.instance_category_matrix = np.zeros([self.num_instances, self.num_categories], int)
        self.instance_instance_matrix = np.zeros([self.num_instances, self.num_instances], int)
        self.instance_index_dict = {}
        self.instance_category_index_dict = {}

        for i in range(self.num_instances):
            self.instance_index_dict[self.instance_list[i]] = i
            instance1 = self.instance_list[i]
            category1 = self.instance_category_dict[instance1]
            j = self.category_index_dict[category1]
            self.instance_category_matrix[i, j] = 1
            self.category_instance_list_dict[category1].append(instance1)

            self.instance_category_index_dict[instance1] = j

            for j in range(self.num_instances):

                instance2 = self.instance_list[j]
                category2 = self.instance_category_dict[instance2]
                if category1 == category2:
                    self.instance_instance_matrix[i, j] = 1

    def remove_instances(self, instance_list):
        for instance in instance_list:
            del self.instance_category_dict[instance]
        self.create_from_instance_category_dict()

    def set_instance_feature_matrix(self, data_matrix, data_index_dict):
        #  embedding matrix vocab size x embedding size 8192x128
        #  num_instances x embedding size  700x128

        self.instance_size = data_matrix.shape[1]
        self.instance_feature_matrix = np.zeros([self.num_instances, self.instance_size])

        for i in range(self.num_instances):

            instance = self.instance_list[i]
            if instance in data_index_dict:
                embedding = data_matrix[data_index_dict[instance], :]
                self.instance_feature_matrix[i, :] = embedding
            else:
                raise ValueError(f"Instance {instance} not in data matrix")

    def create_xy_lists(self):

        self.x_index_list = []
        self.x_list = []
        self.y_list = []

        for i in range(self.num_instances):
            instance = self.instance_list[i]
            category = self.instance_category_dict[instance]
            category_index = self.category_index_dict[category]
            instance_data = self.instance_feature_matrix[i, :]
            self.x_index_list.append(i)
            self.x_list.append(instance_data)
            self.y_list.append(category_index)

    def create_train_test_datasets(self, test_split=0.0):
        # Shuffle the input_data
        dataset_size = self.num_instances
        indices = list(range(dataset_size))
        random.shuffle(indices)
        shuffled_data = [self[i] for i in indices]

        all_shuffled_data = []
        for i in range(len(shuffled_data)):
            all_shuffled_data.append((indices[i], shuffled_data[i][0], shuffled_data[i][1]))

        # Splitting the input_data
        test_size = int(dataset_size * test_split)
        train_size = dataset_size - test_size
        train_dataset, test_dataset = random_split(all_shuffled_data, [train_size, test_size])

        return train_dataset, test_dataset


