'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all libraries
import numpy as np
import pandas as pd
import sys
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class TKRKNN:
    def __init__(self, binary_columns : list = None, seed : 'int > 0' = 42, k : 'int > 0' = 5) -> None:
        '''
            The constructor of TRKNN.
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param seed: int > 0, default = 42
            The seed for the random number generator.
        :param k: int > 0, default = 5
            The number of neighbours used by the knn algorithm.
        '''
        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns
        self.__seed = seed
        self.__k = k
        np.random.seed(self.__seed)
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

    def __predict_knn(self, sample, cls):
        '''
            The knn algorithm in one function.
        :param sample: 1-array
            The array for which we should find the nearest neighbours.
        :param cls: object
            The class that we want to query.
        :return: array
            The indexes of the samples that are the nearest to the sample.
        '''
        distances = []
        for x in self.df[self.df[self.target] == cls][self.X_columns].values:
            distances.append(np.linalg.norm(x - sample, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]
        return predicted_index

    def balance(self, df : 'pd.DataFrame', target : str):
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

        # Getting the sample set of the minority class.
        minority_set = self.df[self.df[self.target] == minority_class][self.X_columns].values

        # Defining the empty nearest neighbours array
        nnarray = []

        # Searching for k nearest neighbours for every sample in the minority class.
        for x in minority_set:
            nnarray.append(self.__predict_knn(x, minority_class))

        # Computing the number of samples to generate.
        samples_to_generate = len(self.df[self.df[self.target] == majority_class]) - len(self.df[self.df[self.target] == minority_class])

        # Defining the empty list for synthetic data.
        self.synthetic_data = []

        # Generating of the synthetic data.
        while samples_to_generate != 0:
            # Randomly selecting a sample from minority set.
            nn = np.random.randint(low=0, high=self.__k, size=1)
            random_index = np.random.randint(low=0, high=len(minority_set), size=1)[0]
            self.synthetic_data.append(minority_set[random_index])

            # Randomly change the attributes of the selected sample.
            for attr_index in range(len(self.X_columns)):
                dif = minority_set[nnarray[random_index][nn], attr_index] - minority_set[random_index, attr_index]
                gap = np.random.uniform(size=1)
                self.synthetic_data[-1][attr_index] = minority_set[random_index, attr_index] + gap * dif
            samples_to_generate -= 1

        # Replacing infinity values with minimal and maximal float python values.
        self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data).astype(float))

        # Creating the synthetic data frame.
        self.synthetic_df = pd.DataFrame(self.synthetic_data, columns=self.X_columns)

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        # Adding the target column.
        self.synthetic_df.loc[:, self.target] = minority_class
        new_df = pd.concat([self.df, self.synthetic_df], axis=0)
        return new_df