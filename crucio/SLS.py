'''
Created with love by Sigmoid

@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from random import randrange
import math
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class SLS:

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

    def balance(self, df : pd.DataFrame, target : str):
        '''
            Reducing the dimensionality of the data
        :param df: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The target name of the value that we have to predict
        '''
        
        unique = df[target].unique()

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        self.df = df.copy()
        self.target = target

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]
        self.synthetic_data = []
        self.synthetic_final_df = self.df.copy()

        classes_nr_samples = []
        new_sample = []

        #check the minority and majority class
        for cl in unique:
            classes_nr_samples.append(len(df[df[target]==cl]))
            majority_class_nr_samples = max(classes_nr_samples)
        majority_class = unique[np.argmax(classes_nr_samples)]

        #minority and majority samples
        minority_classes = [cl for cl in unique if cl!=majority_class]
        minority_classes_samples = [cl for i,cl in enumerate(classes_nr_samples) if i!=np.argmax(classes_nr_samples)]

        #for every class perform oversample
        for i,minority_class in enumerate(minority_classes):

            #difference in samples between minority and majority class
            difference = majority_class_nr_samples-minority_classes_samples[i]

            #minority sample
            minority_samples = self.df[self.df[target] == minority_class][self.X_columns].values
            new_examples_nr=0
            for minority_sample in minority_samples:

                #check if already have needed number of samples
                if new_examples_nr == difference:
                    break

                #randomly select one of the neares neighbors of the sample
                minority_nearest_neighbours = self.__get_k_neighbours(minority_sample,minority_samples)
                index = randrange(len(minority_nearest_neighbours))
                neighbour_index = minority_nearest_neighbours[index]
                selected_neighbour = minority_samples[neighbour_index]

                #check the number of positive neighbour around sample an its neighbor which is safe level for itself
                positive_instances_example = self.__get_k_neighbours_positive_number(minority_sample,minority_class)
                positive_instances_neighb = self.__get_k_neighbours_positive_number(selected_neighbour,minority_class)

                #calculate the ration by dividing the save level of sample and save level of neighbor
                if positive_instances_neighb!=0:
                    ratio = positive_instances_example/positive_instances_neighb
                else :
                    ratio = -1

                #if ration == -1 do not sample a new instance
                if ratio == -1 and positive_instances_example == 0:
                    continue

                #sample new instance
                else :

                    #depending on ration, chose where to create new neigbor
                    #gap is the number that shows how close synthetic sample would be to our minority sample
                    for j in range(len(minority_sample)):
                        if ratio == -1 and positive_instances_example!=0:
                            gap = 0
                        elif ratio == 1:
                            gap = random.random()
                        elif ratio > 1:
                            gap = random.uniform(0,1/ratio)
                        elif ratio<1: 
                            gap = random.uniform(1-ratio,1)
                        dif = selected_neighbour[j]-minority_sample[j]

                        #adding new example to df
                        new_sample.append(minority_sample[j]+gap*dif)
                    self.synthetic_data.append(new_sample)
                    new_examples_nr+=1
                    new_sample = []

            self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data))
            self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)
            self.synthetic_data = []
            self.synthetic_df.loc[:, target] = minority_class

            self.synthetic_final_df = pd.concat([self.synthetic_df,self.synthetic_final_df],axis=0)  

        self.synthetic_df = self.synthetic_final_df.copy()
        
        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        #return new df
        return self.synthetic_df
    



    def __get_k_neighbours(self,example,minority_samples):
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        '''

        distances = []
        for x in minority_samples:
            distances.append(np.linalg.norm(x - example, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]
        
        return predicted_index

    def __get_k_neighbours_positive_number(self,example,minority_class):
        '''
            KNN, getting all positive nearest neigbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        '''

        distances = []

        for x in self.df[self.X_columns].values:
            distances.append(np.linalg.norm(x - example, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]
        count = 0

        for i in predicted_index:
            if self.df[self.target].values[i]==minority_class:
                count+=1
        return count