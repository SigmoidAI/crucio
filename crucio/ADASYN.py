'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all libraries
import numpy as np
import pandas as pd
import random
import sys
from math import floor
from .erorrs import NotBinaryData, NoSuchColumn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class ADASYN:
    def __init__(self, binary_columns : list = None, beta : "float < 1" = 1.0, k : "int > 0" = 5, seed : int = 42) -> None:
        '''
            The constructor of the ADASYN algorithm.
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param beta: float <= 1, default = 1.0
            The ration of minority : majority data desired after ADASYN.
        :param k: int > 0, default = 5
            The number of neighbours used by the knn algorithm.
        :param seed: int, default = 42
            The seed for random number generator.
        '''
        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns
        self.__beta = beta
        self.__k = k
        self.__seed = seed
        np.random.seed(self.__seed)
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

        # Getting the total number of minority samples to generate.
        G = int((classes_frequency[majority_class] - classes_frequency[minority_class]) * self.__beta)

        # Getting the set of the minority samples.
        minority_samples = self.df[self.df[target] == minority_class][self.X_columns].values

        # Generating the r matrix - the k indexes of the nearest neighbours.
        r = np.array([])
        self.neighbourhood = []
        for minority_sample in minority_samples:
            predicted_indexes = self.__predict_knn(minority_sample)
            r = np.append(r, len(self.df[(self.df.index.isin(predicted_indexes) & (self.df[self.target] == majority_class))]) / self.__k)
            self.neighbourhood.append(predicted_indexes)

        # Normalizing the r array
        r = r / np.sum(r)

        # Calculating the amount of synthetic examples to generate per neighbourhood.
        G = r * G

        # Generating the synthetic data.
        self.synthetic_data = []
        for i in range(len(G)):
            for _ in range(floor(G[i])):
                choices = self.df.iloc[self.neighbourhood[i], :][self.df[self.target] == minority_class][self.X_columns].values
                if len(choices) < 2:
                    continue
                choices = choices[
                    np.random.randint(len(choices), size=2)]
                s = choices[0] + (choices[1] - choices[0]) * random.uniform(0, 1)
                self.synthetic_data.append(s)

        # Replacing infinity values with minimal and maximal float python values.
        self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data).astype(float))

        # Creating the synthetic data frame
        self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        # Adding the target column
        self.synthetic_df.loc[:, self.target] = minority_class
        new_df = pd.concat([self.df, self.synthetic_df], axis=0)
        return new_df

    def __predict_knn(self, sample):
        '''
            The knn algorithm in one function.
        :param sample: 1-array
            The array for which we should find the nearest neighbours.
        :return: array
            The indexes of the samples that are the nearest to the sample.
        '''
        distances = []
        for x in self.df[self.X_columns].values:
            distances.append(np.linalg.norm(x - sample, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]
        return predicted_index