'''
Created with love by Sigmoid

@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from random import randrange
import statistics
import math
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


class SMOTENC:


    def __init__(self,k: "int > 0" = 5, seed: float = 42, binary_columns : list = None) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param seed: int, default = 42
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

    def balance(self, df : pd.DataFrame, target : str)-> pd.DataFrame:
        '''
            Balance all minority classes to the number of majority class instances
        :param df: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The target name of the value that we have to predict
        '''

        #check for binary
        unique = df[target].unique()

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")


        self.df = df.copy()
        self.synthetic_final_df = self.df.copy()

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]

        classes_nr_samples = []

        #Find the majority class and creating nr of minority calss samples list
        for cl in unique:
            classes_nr_samples.append(len(df[df[target]==cl]))
            majority_class_nr_samples = max(classes_nr_samples)
        majority_class = unique[np.argmax(classes_nr_samples)]

        minority_classes = [cl for cl in unique if cl!=majority_class]
        minority_classes_samples = [cl for i,cl in enumerate(classes_nr_samples) if i!=np.argmax(classes_nr_samples)]         

        #check for categorical data i 
        not_categorical_columns = [] 
        for column in self.df.columns:
            if column!=target and df[column].values.dtype!=object:
                not_categorical_columns.append(column)

        for i,minority_class in enumerate(minority_classes):

            #find difference between nr samples of minority and mojority class
            difference = majority_class_nr_samples-minority_classes_samples[i]

            #set minority class for specific class
            minority_df = self.df[self.df[target] == minority_class][self.X_columns]
            minority_samples = self.df[self.df[target] == minority_class][self.X_columns].values
            self.synthetic_data = []
            std_list = []

            #add columns for which calculate std
            for col in not_categorical_columns:
                std_list.append(np.std(minority_df[col].values))
            std_list.sort()

            #calculate mean to add for every minority class
            self.median = statistics.median(std_list)

            for _ in range(difference):

                #random example from minority class
                index = randrange(len(minority_samples))
                example = minority_samples[index]

                #select k neighbouors from this example
                neighbours_indexes = self.__get_k_neighbours(example,minority_samples) 

                #select random neighbour
                index = randrange(len(neighbours_indexes))
                selected_neighbour = neighbours_indexes[index]
                selected_neighbour = minority_samples[selected_neighbour]

                #select random point between example and neighbour
                alpha = random.random()
                new_example = []

                #depending if categorical then the most common value from neighbours, otherwise, simple SMOTE
                for feature,_ in enumerate(example):
                    if type(example[feature])!=str:
                        new_example.append(example[feature]+alpha*(selected_neighbour[feature]-example[feature]))
                    else: 
                        neighbours = pd.DataFrame(minority_samples,columns = self.X_columns )
                        feature_list = neighbours[self.X_columns[feature]].values
                        neighbours_features = []
                        for n in neighbours_indexes:
                            neighbours_features.append(feature_list[n])
                        new_example.append(self.__most_common(neighbours_features))

                #add it to df
                self.synthetic_data.append(new_example)
            self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data))
            self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)
            
            print(self.synthetic_df.info())
            self.synthetic_data = []
            self.synthetic_df.loc[:, target] = minority_class

            #to all dataset add new rows
            self.synthetic_final_df = pd.concat([self.synthetic_df,self.synthetic_final_df],axis=0)

        
        self.synthetic_df = self.synthetic_final_df.copy()
        
        #convert back to float noncategorical columns
        for col in not_categorical_columns:
            self.synthetic_df[col] = self.synthetic_df[col].astype(float)

        # Rounding binary columns if needed.      
        if self.__binarize:
            self.__to_binary()

        #return new df
        return self.synthetic_df
    
    def __get_k_neighbours(self,example,minority_sample):
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        '''
        distances = []

        for x in minority_sample:
            dist=0
            for feature,_ in enumerate(x):
                if type(x[feature])!=str:
                    dist+=(x[feature]-example[feature])**2
                else:
                    dist+=self.median**2
            distances.append(math.sqrt(dist))

        predicted_index = np.argsort(distances)[1:self.__k + 1]
        return predicted_index

    def __most_common(self,neighbours_features):
        '''
        Getting the most common categorical value for K nearest neighbours
        '''

        word_counter = {}
        for word in neighbours_features:
            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1

        popular_words = sorted(word_counter, key = word_counter.get, reverse = True)

        return popular_words[0]