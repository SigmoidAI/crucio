'''
Created with love by Sigmoid

@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from random import randrange
from .SMOTE import SMOTE
from sklearn.mixture import GaussianMixture
from .erorrs import NotBinaryData, NoSuchColumn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class SCUT:


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
            self.__binary_columns = None
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

        #get unique values from target column
        unique = df[target].unique()

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        self.target= target
        self.df = df.copy()

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]

        class_samples = []
        for clas in unique:
            class_samples.append(self.df[self.df[self.target]==clas][self.X_columns].values)


        classes_nr_samples = []
        for clas in unique:
            classes_nr_samples.append(len(self.df[self.df[self.target]==clas]))

        #getting mean number of samples of all classes
        mean = np.mean(classes_nr_samples)

        #undersampling by SCUT algorithm
        for i,class_sample in enumerate(class_samples):

            #cheching for
            if classes_nr_samples[i]>mean:
                clusters = []
                selected_samples = []

                #getting nr of samples to take
                difference = classes_nr_samples[i] - mean

                #clusster every class
                gmm = GaussianMixture(3)
                gmm.fit(class_sample)

                #getting predictions
                labels = gmm.predict(class_sample)
                clusters_df = pd.DataFrame(np.array(class_sample), columns=self.X_columns)
                clusters_df.loc[:, target] = labels 
                unique_clusters = clusters_df[self.target].unique()

                #Selecting random samples from every cluster, and repeat untill we get the "difference" that we needed
                for clas in unique_clusters:

                    #clusster sets
                    clusters.append(clusters_df[clusters_df[self.target]==clas].values)

                while difference != 0:
                    for cluster in clusters:

                        #selecting random sample and add it to list
                        index = randrange(len(cluster))
                        example = cluster[index]
                        selected_samples.append(example)
                        difference-=1
                        if difference == 0:
                            break

                #create the new dataset including selected samples
                cluster_df = pd.DataFrame(np.array(selected_samples), columns=self.df.columns)
                cluster_df.loc[:, self.target] = unique[i]##
                if i==0:
                    self.new_df = cluster_df
                else:
                    self.new_df = pd.concat([cluster_df,self.new_df],axis=0)
            else:
                
                cluster_df = pd.DataFrame(np.array(class_sample), columns=self.X_columns)
                cluster_df.loc[:, self.target] = unique[i]##
                if i==0:
                    self.new_df = cluster_df
                else:
                    self.new_df = pd.concat([cluster_df,self.new_df],axis=0)


        #oversampling using SMOTE
        smot = SMOTE(binary_columns = self.__binary_columns)

        #getting finall dataset
        self.new_df = smot.balance(self.new_df,self.target)

        #return new dataset
        return self.new_df