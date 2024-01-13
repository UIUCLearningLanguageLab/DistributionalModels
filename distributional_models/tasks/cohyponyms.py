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

        self.guess_accuracy_df = None
        self.guess_accuracy_df_list = []
        self.target_balanced_accuracy_df = None
        self.target_accuracy_df = None
        self.best_threshold = None

        self.target_ba_df = None
        self.category_ba_df = None
        self.threshold_ba_df = None
        self.best_target_ba_df = None
        self.best_category_ba_df = None
        self.guess_accuracy_best_df = None
        self.overall_target_mean = None
        self.overall_category_mean = None

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
        self.create_results_df()
        self.compute_ba()
        self.took = time.time() - start_time

    def create_results_df(self):

        threshold_hit_list = np.zeros([self.num_thresholds])
        threshold_cr_list = np.zeros([self.num_thresholds])
        threshold_yes_list = np.zeros([self.num_thresholds])
        threshold_no_list = np.zeros([self.num_thresholds])

        results = []
        for i in range(self.num_thresholds):
            threshold = self.threshold_list[i]

            for j in range(self.categories.num_instances):
                target1 = self.categories.instance_list[j]
                category1 = self.categories.instance_category_dict[target1]

                for k in range(self.categories.num_instances):

                    if j != k:
                        target2 = self.categories.instance_list[k]
                        category2 = self.categories.instance_category_dict[target2]
                        score = self.similarity_matrix[j, k]

                        if -1 > score < 1:
                            raise ValueError(f"ERROR: Matrix score {j}{k}:{score} out of bounds 0-1")

                        if score > threshold:
                            guess = 1
                        else:
                            guess = 0

                        if category1 == category2:
                            actual = 1
                            threshold_yes_list[i] += 1
                        else:
                            actual = 0
                            threshold_no_list[i] += 1

                        if guess == actual:
                            correct = 1
                            if guess == 1:
                                sd_result = "hit"
                                threshold_hit_list[i] += 1
                            else:
                                sd_result = "cr"
                                threshold_cr_list[i] += 1
                        else:
                            correct = 0
                            if guess == 0:
                                sd_result = "miss"
                            else:
                                sd_result = "fa"

                        data = [threshold,
                                target1,
                                target2,
                                category1,
                                category2,
                                j,
                                k,
                                score,
                                guess,
                                actual,
                                correct,
                                sd_result]
                        results.append(data)

        self.guess_accuracy_df = pd.DataFrame(results, columns=['threshold',
                                                                'target1',
                                                                'target2',
                                                                'category1',
                                                                'category2',
                                                                'target1_index',
                                                                'target2_index',
                                                                'score',
                                                                'guess',
                                                                'actual',
                                                                'correct',
                                                                'sd_result'])

        hit_rate = threshold_hit_list / threshold_yes_list
        cr_rate = threshold_cr_list / threshold_no_list
        ba = (hit_rate + cr_rate) / 2
        best_threshold_index = np.argmax(ba)
        self.best_threshold = self.threshold_list[best_threshold_index]

        if self.only_best_threshold:
            self.guess_accuracy_df = self.guess_accuracy_df[self.guess_accuracy_df['threshold'] == self.best_threshold]

    def compute_ba(self):
        self.target_accuracy_df = self.guess_accuracy_df.groupby(['threshold',
                                                                  'target1',
                                                                  'category1',
                                                                  'actual'])['correct'].agg(['mean',
                                                                                             'std',
                                                                                             'count']).reset_index()

        self.target_accuracy_df.columns = ['threshold', 'target', 'category', 'actual', 'mean_correct',
                                           'stdev_correct', 'n']

        # Some other grouping operation
        self.target_ba_df = self.target_accuracy_df.groupby(['threshold', 'target', 'category'])[
            'mean_correct'].mean().reset_index()
        self.target_ba_df.columns = ['threshold', 'target', 'category', 'ba']

        # getting category means
        self.category_ba_df = self.target_ba_df.groupby(['threshold', 'category'])['ba'].agg(
            ['mean', 'std', 'count']).reset_index()
        self.category_ba_df.columns = ['threshold', 'category', 'mean_ba', "std_ba", 'n']
        self.category_ba_df['stderr_ba'] = self.category_ba_df['std_ba'] / np.sqrt(self.category_ba_df['n'])

        if self.only_best_threshold:
            self.best_category_ba_df = self.category_ba_df
            self.best_target_ba_df = self.target_ba_df
        else:
            # determining the best threshold
            self.threshold_ba_df = self.category_ba_df.groupby(['threshold'])['mean_ba'].mean().reset_index()
            self.threshold_ba_df.columns = ['threshold', 'mean_ba']
            # # Get the index of the row with the highest value of mean_ba in threshold_ba_df
            max_mean_ba_index = self.threshold_ba_df['mean_ba'].idxmax()
            max_mean_ba_threshold = self.threshold_ba_df.loc[max_mean_ba_index, 'threshold']

            # filtering by best threshold
            self.best_target_ba_df = self.target_ba_df[
                self.target_ba_df['threshold'] == max_mean_ba_threshold].reset_index()

            self.best_category_ba_df = self.best_target_ba_df.groupby(['category', 'threshold'])['ba'].agg(
                ['mean', 'std', 'count']).reset_index()
            self.best_category_ba_df.columns = ['category', 'best_overall_threshold', 'ba_mean', 'ba_std', 'n']
            self.best_category_ba_df = self.best_category_ba_df.applymap(lambda x: round_and_check(x, 8))
            self.guess_accuracy_best_df = self.guess_accuracy_df[
                self.guess_accuracy_df['threshold'] == self.best_category_ba_df['best_overall_threshold'].unique()[0]]

        self.overall_target_mean = self.best_target_ba_df['ba'].mean()
        self.overall_category_mean = self.best_category_ba_df['mean_ba'].mean()

    def save_results(self, path):
        self.best_category_ba_df.to_csv(path, index=False)
