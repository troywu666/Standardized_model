# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 04:34:22
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-03-04 21:02:21

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.feature_selection import chi2, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Selector():
	def __init__(self, fea):
		self.fea = fea

	def variance(self, threshold = 0):
		'''
		method可选“var_filter”、“chi2_filter”、“f_clas_filter、“f_reg_filter”、“mutual_clas_filter”、“mutual_reg_filter”三种
		'''
		selector = VarianceThreshold(threshold).fit(self.fea)
		return selector, selector.transform(self.fea)

	def filter(self, y, k, method = 'chi2_filter'):
		if method == 'chi2_filter':
			selector = SelectKBest(chi2, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		elif method == 'f_clas_filter':
			selector = SelectKBest(f_classif, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		elif method == 'f_reg_filter':
			selector = SelectKBest(f_regression, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		elif method == 'mutual_clas_filter':
			selector = SelectKBest(mutual_info_classif, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		elif method == 'mutual_reg_filter':
			selector = SelectKBest(mutual_info_regression, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		else:
			print('The parameters were worng!')

	def wrapper(self, estimator, y, n, step = None):
		selector = RFE(estimator, n, step).fit(self.fea, y)
		return selector, selector.transform(self.fea)
	
	def embedded(self, estimator, y, threshold):
		selector = SelectFromModel(estimator, threshold).fit(self.fea, y)
		return selector, selector.transform(self.fea)

