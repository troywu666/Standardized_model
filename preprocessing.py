# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 06:46:40
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-25 15:54:58

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, Binarizer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from scipy import stats

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

class Transformer():
	def __init__(self, data):
		self.data = data

	def scaler(self, method):
		if method == 'min_max':
			transformer = MinMaxScaler().fit(self.data)
			return transformer, transformer.transform(self.data)
		if method == 'standard':
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
		if method == 'binarizer':
			transformer = Binarizer(threshold).fit(self.data)
			return transformer, transformer.transform(self.data)
		if method == 'ordinal':
			transformer = OrdinalEncoder().fit(self.data)
			return transformer, transformer.transform(self.data)
		else:
			print('The parameters were worng!')

	def box_cox(self, lmbda = None):
		transformer = Box_cox().fit(self.data)
		return transformer, transformer.transform(self.data, lmbda = lmbda)