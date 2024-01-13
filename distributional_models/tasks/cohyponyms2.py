from ..scripts.utils import round_and_check, create_similarity_matrix
import numpy as np
import pandas as pd
import time


class Cohyponyms:

    def __init__(self, categories, num_thresholds=11, only_best_threshold=False, similarity_metric='correlation'):

        self.categories = categories
        self.num_thresholds = num_thresholds
        self.similarity_metric = similarity_metric

        self.similarity_matrix = None
        self.num_thresholds = num_thresholds
        self.only_best_threshold = only_best_threshold

        self.best_threshold = None
        self.overall_target_mean = None
        self.overall_category_mean = None

        self.guess_matrix = None
        self.accuracy_matrix = None

        self.threshold_list = np.linspace(-1, 1, self.num_thresholds)

        self.took = None

        self.run_cohyponym_task()

    def run_cohyponym_task(self):
        start_time = time.time()

        if self.categories.instance_feature_matrix is not None:
            self.similarity_matrix = create_similarity_matrix(self.categories.instance_feature_matrix,
                                                              self.similarity_metric)
        else:
            raise ValueError("Categories object has no instance feature matrix")

        thresholds_column = np.array(self.threshold_list)[:, np.newaxis, np.newaxis]

        rounded_similarity_matrix = np.around(self.similarity_matrix, decimals=6)

        self.guess_matrix = (rounded_similarity_matrix >= thresholds_column).astype(int)

        cohyponym_matrix_reshaped = self.categories.instance_instance_matrix[np.newaxis, :, :]

        self.correct_matrix = (self.guess_matrix == cohyponym_matrix_reshaped).astype(int)

        same_category_mask = self.categories.instance_instance_matrix == 1
        different_category_mask = self.categories.instance_instance_matrix == 0

        self.same_correct_matrix = np.where(same_category_mask, self.correct_matrix, 0)
        self.different_correct_matrix = np.where(different_category_mask, self.correct_matrix, 0)

        self.same_correct_sums = self.same_correct_matrix.sum(axis=(1, 2)) - self.categories.num_instances
        self.different_correct_sums = self.different_correct_matrix.sum(axis=(1, 2))

        same_count_sum = self.categories.instance_instance_matrix.sum()
        self.same_count_sums = same_count_sum - self.categories.num_instances
        self.different_count_sums = self.categories.num_instances**2 - same_count_sum

        self.same_accuracies = self.same_correct_sums/self.same_count_sums
        self.different_accuracies = self.different_correct_sums / self.different_count_sums

        self.balanced_accuracies = (self.same_accuracies + self.different_accuracies) / 2
        self.balanced_accuracy_mean = np.amax(self.balanced_accuracies)
        self.took = time.time() - start_time

        sim_mask = np.triu(np.ones_like(self.similarity_matrix), k=1).astype(bool)
        flattened_sims = self.similarity_matrix[sim_mask]

        cohyponyms_mask = np.triu(np.ones_like(self.categories.instance_instance_matrix), k=1).astype(bool)
        flattened_cohyponyms = self.categories.instance_instance_matrix[cohyponyms_mask]
        self.correlation = np.corrcoef(flattened_sims, flattened_cohyponyms)[0, 1]

    def save_results(self, path):
        self.best_category_ba_df.to_csv(path, index=False)
