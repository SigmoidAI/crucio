'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all libraries
import numpy as np
import pandas as pd
from math import floor
import sys
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class ICOTE:
    def __init__(self, binary_columns : list = None, seed : 'int > 0' = 42) -> None:
        '''
            The constructor of ICOTE.
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param seed: int > 0, default = 42
            The seed for random number generator.
        '''
        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns
        self.__seed = seed
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

        # Splitting the data in features and target arrays.
        X, y = self.df[self.X_columns].values, self.df[self.target].values

        # Getting the min and max arrays of the features.
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        # Normalising the data.
        norm_X = (X - X_min) / (X_max - X_min)

        # Getting the indexes of the minority and majority classes.
        minority_index = np.where(y == minority_class)
        majority_index = np.where(y == majority_class)

        # Getting the number of samples of minority class.
        minority_samples: int = len(minority_index)

        # Defining the empty list with synthetic data.
        self.synthetic_data = []
        while minority_samples <= len(majority_index):
            # Creating clones for every minority sample.
            for i in minority_index[0]:
                clones = [].copy()
                dist = [].copy()
                for j in majority_index[0]:
                    dist.append(np.linalg.norm(norm_X[i] - norm_X[j], ord=2))
                for dist_index in range(len(dist)):
                    if dist[dist_index] == 0:
                        continue
                    for _ in range(floor(dist[dist_index])):
                        clones.append([norm_X[i], dist_index])
                for clone_index in range(len(clones)):
                    alpha = 1 / dist[clones[clone_index][1]]
                    if alpha < 0.05:
                        continue
                    # Creating mutations in clones.
                    clones[clone_index][0] = clones[clone_index][0] + alpha * (clones[clone_index][0] - norm_X[clones[clone_index][1]])
                # Adding the clones to the synthetic data list.
                self.synthetic_data.extend([clone[0] for clone in clones])
                minority_samples += len(clones)
        # Selecting majority - minority samples.
        selected_data = np.random.randint(len(self.synthetic_data), size=(len(majority_index[0]) - len(minority_index[0]), 1))
        self.synthetic_data = np.array(self.synthetic_data)[selected_data.reshape(1, -1)[0], :]

        # Taking the data to the original dimensions.
        for i in range(len(self.synthetic_data)):
            self.synthetic_data[i] = X_min + (X_max - X_min) * self.synthetic_data[i]

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