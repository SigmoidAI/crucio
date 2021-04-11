'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
import sys

from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class MTDF:
    def __init__(self, binary_columns : list = None, seed : 'int > 0' = 42) -> None:
        '''
            The constructor of the MTDF.
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param seed: int > 0, default = 42
            The seed used for random number generator.
        '''
        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns
        self.seed = seed
        np.random.seed(self.seed)

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

        # Calculating the variance vector.
        variance = self.df[self.X_columns].var(axis=0).values

        # Calculating the Uset.
        uset = (self.df[self.X_columns].min(axis=0).values + self.df[self.X_columns].max(axis=0).values) / 2

        # Getting the classes frequency.
        classes_frequency = dict(self.df[target].value_counts())

        # Searching for the class with the biggest frequency.
        max_freq = 0
        for cls in classes_frequency:
            if classes_frequency[cls] > max_freq:
                majority_class = cls
                max_freq = classes_frequency[cls]

        # Getting the name of the minority class.
        minority_class = [cls for cls in classes_frequency if cls != majority_class][0]

        # Computing the lower and upper skewness.
        skewL = len(self.df[df[self.target] == minority_class]) / len(df)
        skewU = len(self.df[df[self.target] == majority_class]) / len(df)

        # Computing the lower and upper limits.
        a = self.df[self.X_columns].min(axis=0).values / 10
        b = self.df[self.X_columns].max(axis=0).values * 10

        # Updating the lower and upper limits.
        a = uset - skewL * np.sqrt(-2 * (variance / len(self.df[self.df[self.target] == minority_class])) * np.log(norm.cdf(a)))
        b = uset - skewU * np.sqrt(-2 * (variance / len(self.df[self.df[self.target] == majority_class])) * np.log(norm.cdf(b)))

        # Calculating the number of elements to generate.
        number_of_elements_to_generate = len(self.df[self.df[self.target] == majority_class]) - len(self.df[self.df[self.target] == minority_class])

        # Generating the synthetic data.
        self.synthetic_data = []

        # Eliminating possible -infinity and +infinity values.
        a = self.__infinity_check(a)
        b = self.__infinity_check(b)

        for _ in range(number_of_elements_to_generate):
            self.synthetic_data.append(
                # Generating arrays with random values between a and b.
                np.array([
                    np.random.uniform(a[i], b[i], 1)[0] for i in range(len(a))
                ])
            )

        # Replacing infinity values with minimal and maximal float python values.
        self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data).astype(float))

        # Creating the synthetic data frame.
        self.synthetic_df = pd.DataFrame(self.synthetic_data, columns=self.X_columns)

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        # Adding the target column to the synthetic data frame.
        self.synthetic_df.loc[:, self.target] = minority_class

        # Concatting the real data frame with the synthetic one.
        new_df = pd.concat([self.df, self.synthetic_df], axis=0)
        return new_df