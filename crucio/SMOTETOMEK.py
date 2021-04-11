'''
Created with love by Sigmoid

@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from random import randrange
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


class SMOTETOMEK:


    def __init__(self,k: "int > 0" = 5, seed: float = 42, binary_columns : list = None) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param seed: intt, default = 42
            seed for random
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        '''

        self.__k = k

        if binary_columns is None:
            self.__binarize = False
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns

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

    def __undersample(self):
        '''
            Undersample the dataset using Tomek links algorithm
        '''

        distances = []
        minority_nearest = []
        majority_nearest = []

        #save distances for minority class
        for example in self.minority_samples:
            distances = []
            for x in self.majority_samples:
                distances.append(np.linalg.norm(x - example, ord=2))
            predicted_index = np.argsort(distances)[0]
            minority_nearest.append(predicted_index)

        #save distances for majority class
        for example in self.majority_samples:
            distances = []
            for x in self.minority_samples:
                distances.append(np.linalg.norm(x - example, ord=2))
            predicted_index = np.argsort(distances)[0]
            majority_nearest.append(predicted_index)

        #find all tomek links
        to_delete = []
        for i in range(len(minority_nearest)):
            if i == majority_nearest[minority_nearest[i]]:
                to_delete.append(minority_nearest[i])

        #delete all Tomek links
        self.majority_samples = np.delete(self.majority_samples,to_delete,axis = 0)

        # Creating new dataset without Tomek Links
        mj = pd.DataFrame(self.majority_samples,columns = self.X_columns)
        mj.loc[:, self.target] = self.majority_class
        mn = pd.DataFrame(self.minority_samples,columns = self.X_columns)
        mn.loc[:, self.target] = self.minority_class
        self.df = pd.concat([mj,mn],axis = 0)


    def balance(self, df : pd.DataFrame, target : str)-> pd.DataFrame:
        '''
            Reducing the dimensionality of the data
        :param df: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The target name of the value that we have to predict
        '''

        #check for binary
        unique = df[target].unique()
        if len(unique)!=2:
            raise NotBinaryData(f"{target} column isn't a binary column")

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        self.target= target
        self.df = df.copy()
        self.X_columns = [column for column in self.df.columns if column != target]

        #check the minority class
        first_class = len(df[df[target]==unique[0]])/len(df[target])
        if first_class > 0.5:
            self.minority_class,self.majority_class = unique[1],unique[0]
        else: 
            self.minority_class,self.majority_class = unique[0],unique[1]

        majority = df[df[target]==self.majority_class]
        minority = df[df[target]==self.minority_class]

        self.minority_samples = self.df[self.df[target] == self.minority_class][self.X_columns].values
        self.majority_samples = self.df[self.df[target] == self.majority_class][self.X_columns].values

        #calling undersample function
        self.__undersample()

        #find difference 
        difference = len(majority)-len(minority)
        self.synthetic_data = []
        for _ in range(difference):

            #random example from minority class
            index = randrange(len(self.minority_samples))
            example = self.minority_samples[index]

            #select k neighbouors from this example
            neighbours_indexes = self.__get_k_neighbours(example) 

            #select random neighbour
            index = randrange(len(neighbours_indexes))
            selected_neighbour = neighbours_indexes[index]
            selected_neighbour = self.minority_samples[selected_neighbour]

            #select random point between example and neighbour
            alpha = random.random()
            new_example = example+alpha*(selected_neighbour-example)

            #add it to df
            self.synthetic_data.append(new_example)

        self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data))
        self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)
        self.synthetic_df.loc[:, target] = self.minority_class
        self.synthetic_df = pd.concat([self.synthetic_df,self.df],axis=0)

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        #return new df
        return self.synthetic_df
    
    def __get_k_neighbours(self,example):
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        '''
        distances = []
        
        for x in self.minority_samples:
            distances.append(np.linalg.norm(x - example, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]

        return predicted_index