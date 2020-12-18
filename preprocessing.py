'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-11 06:46:40
LastEditors: Troy Wu
LastEditTime: 2020-12-12 13:56:09
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, Binarizer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm_notebook

class Box_cox(TransformerMixin):
	def __init__(self):
		self.best_lmbda = 0
	
	def fit(self, data):
		transformed_array, best_lmbda = stats.boxcox(data.values)
		self.best_lmbda = best_lmbda
		return self

	def transform(self, data, lmbda = None):
		if lmbda == None:
			self.lmbda = lmbda
		return stats.boxcox(data.values, lmbda = self.best_lmbda)

class Transformer:
	def __init__(self, data):
		self.data = data

	def scaler(self, method):
		if method == 'min_max':
			transformer = MinMaxScaler().fit(self.data)
			return transformer, transformer.transform(self.data)
		elif method == 'standard':
			transformer = StandardScaler().fit(self.data)
			return transformer, transformer.transform(self.data)
		else:
			print('The parameters were worng!')

	def fillna(self, fill_value = None, strategy = 'mean'):
		'''
		strategy可选'mean', 'median', 'most_frequent', 'constant'，当选为'constant'时，需要另外设定fill_value
		'''
		transformer = SimpleImputer(strategy = strategy, fill_value = fill_value).fit(self.data)
		return transformer, transformer.transform(self.data)

	def encoder(self, method, threshold = None):
		'''
		method有“onehot”、“ordinal”和“binarizer”三种可填方式
		'''
		if method == 'onehot':
			transformer = OneHotEncoder().fit(self.data)
			return transformer, transformer.transform(self.data)
		elif method == 'binarizer':
			transformer = Binarizer(threshold).fit(self.data)
			return transformer, transformer.transform(self.data)
		elif method == 'ordinal':
			transformer = OrdinalEncoder().fit(self.data)
			return transformer, transformer.transform(self.data)
		else:
			print('The parameters were worng!')

	def box_cox(self, lmbda = None):
		transformer = Box_cox().fit(self.data)
		return transformer, transformer.transform(self.data, lmbda = lmbda)

	def johnson(self, standardize = True, copy = True):
		transformer = PowerTransformer(method = 'yeo-johnson', standardize = standardize, copy = copy).fit(self.data)
		return transformer, transformer.transform(self, data)

class Data_Preprocess:
    def __init__(self):
        self.int8_max = np.iinfo(np.int8).max
        self.int8_min = np.iinfo(np.int8).min

        self.int16_max = np.iinfo(np.int16).max
        self.int16_min = np.iinfo(np.int16).min

        self.int32_max = np.iinfo(np.int32).max
        self.int32_min = np.iinfo(np.int32).min

        self.int64_max = np.iinfo(np.int64).max
        self.int64_min = np.iinfo(np.int64).min

        self.float16_max = np.finfo(np.float16).max
        self.float16_min = np.finfo(np.float16).min

        self.float32_max = np.finfo(np.float32).max
        self.float32_min = np.finfo(np.float32).min

        self.float64_max = np.finfo(np.float64).max
        self.float64_min = np.finfo(np.float64).min
    
    def __get_type(self, min_val, max_val, types):
        if types == 'int':
            if max_val <= self.int8_max and min_val >= self.int8_min:
                return np.int8
            elif max_val <= self.int16_max <= max_val and min_val >= self.int16_min:
                return np.int16
            elif max_val <= self.int32_max and min_val >= self.int32_min:
                return np.int32
            return None

        elif types == 'float':
            if max_val <= self.float16_max and min_val >= self.float16_min:
                return np.float16
            if max_val <= self.float32_max and min_val >= self.float32_min:
                return np.float32
            if max_val <= self.float64_max and min_val >= self.float64_min:
                return np.float64
            return None
    def memory_process(self, df):
        init_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('Original data occupies {} GB memory.'.format(init_memory))
        df_cols = df.columns

          
        for col in tqdm_notebook(df_cols):
            try:
                if 'float' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self.__get_type(min_val, max_val, 'float')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
                elif 'int' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self.__get_type(min_val, max_val, 'int')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
            except:
                print(' Can not do any process for column, {}.'.format(col)) 
        afterprocess_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('After processing, the data occupies {} GB memory.'.format(afterprocess_memory))
        return df