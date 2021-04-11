'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all libraries
import numpy as np
import pandas as pd
from functools import reduce
from operator import or_
from numba import njit
import random
import sys

from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

class MWMOTE:
    def __init__(self, binary_columns : list = None, k1 : 'int > 1' = 5, k2 : 'int > 1' = 5, k3 : 'int > 1' = 5, Cth : 'int > 0' = 5, CMAX : 'int > 0' = 2, M : 'int' = 5, seed : 'int > 0' = 42):
        '''
            The constructor of the MWMOTE.
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param k1: int > 1, default = 5
            Number of neighbors used for predicting noisy minority class samples.
        :param k2: int > 1, default = 5
            Number of majority neighbors used for constructing informative minority set.
        :param k3: int > 1, default = 5
            Number of minority neighbors used for constructing informative minority set.
        :param Cth: int > 0, default = 5
            The threshold value of the closeness factor.
        :param CMAX: int > 0, default = 5
            Used in smoothing and re-scaling the values of different scaling factors.
        :param M: int, default = 5
            The number of clusters.
        :param seed: int > 0, default = 42
            The seed for the random number generator.
        '''
        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns
        self.__k1 = k1
        self.__k2 = k2
        self.__k3 = k3
        self.__Cth = Cth
        self.__CMAX = CMAX
        self.__M = M
        self.__seed = seed
        random.seed(self.__seed)

    def __to_binary(self) -> None:
        '''
            If the :param binary_columns: is set to True then the intermediate values in binary columns will be rounded.
        '''
        for column_name in self.__binary_columns:
            serie = self.synthetic_df[column_name].values
            threshold = (self.df[column_name].max() + self.df[column_name].min()) / 2
            for i in range(len(serie)):
                if serie[i] >= threshold:
                    serie[i] = self.df[column_name].max()
                else:
                    serie[i] = self.df[column_name].min()
            self.synthetic_df[column_name] = serie

    def __infinity_check(self, matrix : 'np.array') -> 'np.array':
        '''
            This function replaces the infinity and -infinity values with the minimal and maximal float python values.
        :param matrix: 'np.array'
            The numpy array that was generated my the algorithm.
        :return: 'np.array'
            The numpy array with the infinity replaced values.
        '''
        matrix[matrix == -np.inf] = sys.float_info.min
        matrix[matrix == np.inf] = sys.float_info.max
        return matrix

    def __predict_knn(self, sample, k, cls = 'auto'):
        '''
            The knn algorithm in one function.
        :param sample: 1-array
            The array for which we should find the nearest neighbours.
        :param k: int
            The number of neighbours.
        :param cls: str, default = 'auto'
            The class that we are interested in:
                If set to 'auto' then all classes are taken
                Else is taken the class passed.
        :return: array
            The indexes of the samples that are the nearest to the sample.
        '''
        if cls == 'auto':
            df = self.df[self.X_columns].values
        else:
            df = self.df[self.df[self.target] == cls][self.X_columns].values
        distances = []
        for x in df:
            distances.append(np.linalg.norm(x - sample, ord=2))
        predicted_index = np.argsort(distances)[1:k + 1]
        return predicted_index

    def __closeness_factor(self, y, x):
        '''
            The closeness factor function.
        :param y:
            The target class.
        :param x:
            The sample.
        :return:
            The closeness factor.
        '''
        dist = np.linalg.norm(y - x, ord=2)
        dist = dist / len(y)
        f = lambda x : x if x <= self.__Cth else self.__Cth
        return f(1 / dist) * self.__CMAX / self.__Cth

    def __density_factor(self, y, x):
        '''
            The density factor function.
        :param y:
            The target class.
        :param x:
            The sample.
        :return:
            The density factor.
        '''
        return self.__closeness_factor(y, x) / np.sum([self.__closeness_factor(y, q) for q in self.informative_minority_set])

    def __information_weight(self, y, x):
        '''
            The information weight function.
        :param y:
            The target class.
        :param x:
            The sample.
        :return:
            The information weight.
        '''
        return self.__closeness_factor(y, x) * self.__density_factor(y, x)

    def __dbscan(self, minority_class_sample):
        '''
            The BDSCAN fit function.
        :param minority_class_sample:
            The set of minority class samples
        :return: list
            The list of clusters of every sample from the set.
        '''
        clusters = np.arange(len(minority_class_sample))
        while len(set(clusters)) > self.__M:
            first = np.random.randint(0, len(clusters))
            second = np.random.randint(0, len(clusters))
            if first == second:
                continue
            min_distance = np.linalg.norm(minority_class_sample[first] - minority_class_sample[second], ord=2)
            closest_pair = [first, second]
            for i in range(len(minority_class_sample)):
                for j in range(len(minority_class_sample)):
                    if clusters[i] == clusters[j]:
                        continue
                    elif min_distance >= np.linalg.norm(minority_class_sample[i] - minority_class_sample[j], ord=2) and np.linalg.norm(minority_class_sample[i] - minority_class_sample[j], ord=2) != 0:
                        min_distance = np.linalg.norm(minority_class_sample[i] - minority_class_sample[j], ord=2)
                        closest_pair = [clusters[i], clusters[j]]
            to_replace = closest_pair[0]
            clusters[clusters == to_replace] = closest_pair[1]
        return clusters

    def __pridict_dbscan(self, sample, minority_class_sample, clusters):
        '''
            The DBSCAN predict function.
        :param sample: array
            The array which we must cluster.
        :param minority_class_sample: 2-d array
            The minority set.
        :param clusters: array
            The array with clusters for every sample.
        :return: int
            The cluster of the sample.
        '''
        dist = []
        for x in minority_class_sample:
            dist.append(np.linalg.norm(x - sample, ord=2))
        return clusters[np.argsort(np.array(dist))[0]]

    def balance(self, df : pd.DataFrame, target : str):
        '''
            The balance function.
        :param df: pd.DataFrame
            The pandas Data Frame to apply the balancer.
        :param target: str
            The name of the target column.
        :return: pd.DataFrame
            A pandas Data Frame
        '''

        # Creating an internal copy of the data frame.
        self.df = df.copy()
        self.target = target

        # Checking if the target string based t algorithm is present in the data frame.
        if target not in self.df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        # Checking if the target column is a binary one.
        if len(self.df[target].unique()) != 2:
            raise NotBinaryData(f"{target} column isn't a binary column")

        # Getting the column names that are not the target one.
        self.X_columns = [column for column in self.df.columns if column != target]

        # Getting the class frequencies.
        classes_frequency = dict(self.df[target].value_counts())

        # Searching for the class with the biggest frequency.
        max_freq = 0
        for cls in classes_frequency:
            if classes_frequency[cls] > max_freq:
                majority_class = cls
                max_freq = classes_frequency[cls]

        # Getting the name of the minority class.
        minority_class = [cls for cls in classes_frequency if cls != majority_class][0]

        # Getting the data for the minority class set.
        minority_set = self.df[self.df[self.target] == minority_class][self.X_columns].values

        # Defining the empty filtered minority set.
        filtered_minority_set = []

        # Constructing the filtered minority set
        for i in range(len(minority_set)):
            neightbours_index = self.__predict_knn(minority_set[i], self.__k1)
            if minority_class not in dict(self.df.iloc[neightbours_index, :][self.target].value_counts()).values():
                filtered_minority_set.append(i)

        # Defining the empty nearest neighbours list.
        NearestNeighbours = []

        # Searching for the k2 nearest neighbours for every sample in filtered minority set
        for i in filtered_minority_set:
            NearestNeighbours.append(self.__predict_knn(self.df.iloc[i, :][self.X_columns].values, self.__k2, majority_class))

        # Converting the NearestNeighbours into a list of sets.
        NearestNeighbours = [set(element.tolist()) for element in NearestNeighbours]

        # Getting the border line majority set
        border_line_majority_set = list(reduce(or_, NearestNeighbours))

        # Clearing the NearestNeighbours list
        NearestNeighbours = []

        # Getting the k3 nearest neighbours for every sample from border line majority set
        for x in border_line_majority_set:
            NearestNeighbours.append(self.__predict_knn(x, self.__k3, minority_class))

        # Converting the NearestNeighbours into a list of sets.
        NearestNeighbours = [set(element.tolist()) for element in NearestNeighbours]

        # Getting the informative minority set.
        self.informative_minority_set = np.array(list(reduce(or_, NearestNeighbours)))

        # Getting the selection weights.
        selection_weights = np.array([np.sum([self.__information_weight(self.df[self.df[self.target] == majority_class].values[y],
                                                                        self.df[self.df[self.target] == minority_class].values[x])
                                              for y in border_line_majority_set]) for x in self.informative_minority_set])
        # Getting the selection probabilities.
        selection_probability = selection_weights / np.sum(selection_weights)

        # Getting the clusters from dbscan.
        clusters = self.__dbscan(minority_set)

        # Defining the empty list with synthetic data.
        self.synthetic_data = []

        # Getting the number of new samples to generate.
        N = len(self.df[self.df[self.target] == majority_class]) - len(self.df[self.df[self.target] == minority_class])

        # Generating the new samples.
        for i in range(N):
            # Getting the first random sample from minority set.
            first = self.df.iloc[np.random.choice(self.informative_minority_set, 1, p = selection_probability), :][self.X_columns].values[0]

            # Getting the cluster of the first selected sample.
            cluster = self.__pridict_dbscan(first, minority_set, clusters)

            if len(clusters[clusters == cluster]) >= 2:
                # Getting the second random sample from the same cluster.
                second = self.df.iloc[np.random.choice(
                    self.informative_minority_set[
                        list(index(clusters, cluster))], 1),
                         :][self.X_columns].values[0]
            else:
                second = self.df.iloc[self.__predict_knn(first, self.__k1, minority_class), :][self.X_columns].values[0]

            # Appending the generated point to the synthetic data.
            self.synthetic_data.append(first + random.uniform(0, 1) * (second - first))

        # Replacing infinity values with minimal and maximal float python values.
        self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data).astype(float))

        # Creating the synthetic data frame.
        self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        # Adding the target column.
        self.synthetic_df.loc[:, self.target] = minority_class
        new_df = pd.concat([self.df, self.synthetic_df], axis=0)
        return new_df