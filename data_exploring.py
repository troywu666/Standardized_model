# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 21:29:47
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-21 16:05:14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sns

class Explore():
	def __init__(self, df):
		self.df = df

	def describe_num(self):
		df_num = self.df.select_dtypes(exclude = [np.object])
		des_num = df_num.describe(percentiles = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.9, 0.95, 0.99]).T.\
		assign(**{'偏度': np.array([df_num[col].skew() for col in df_num.columns]),
	        '峰度': np.array([df_num[col].kurt() for col in df_num.columns]),
	        'na_counts': np.array([df_num[col].isnull().sum() for col in df_num.columns]),
	        'na_pct': np.array([df_num[col].isnull().sum() / df_num.shape[0] for col in df_num.columns])}).sort_index(axis = 1)
		return des_num[['min', '12.5%', '25%', '37.5%', '50%', '62.5%', '75%', '87.5%', '90%', '95%', '99%', 'max', \
			'count', '偏度', '峰度', 'mean', 'std', 'na_pct', 'na_counts']]
	
	def describe_obj(self):
		df_obj = self.df.select_dtypes(include = [np.object])
		des_obj = df_obj.describe().T.assign(
			**{'na_counts': np.array([self.df[col].isnull().sum() for col in df_obj.columns]),
	           'na_pct': np.array([self.df[col].isnull().sum() / df_obj.shape[0] for col in df_obj.columns])}).sort_index(axis = 1)
		return des_obj

	def corr_and_plot(self, figsize = (32, 18)):
		plt.subplots(figsize = figsize)
		sns.heatmap(self.df.corr().round(2), annot = True, cmap='RdBu')
		#plt.show()

	def distplot(self):
		df_num = self.df.select_dtypes(exclude = [np.object]).dropna()
		f, ax = plt.subplots(len(df_num.columns), 1, figsize = (10, len(df_num.columns) * 6))
		for n, col in enumerate(df_num.columns):
			sns.distplot(df_num[col], ax = ax[n])
			plt.title(col)
		#plt.show()

	def pairplot(self, vars = None, hue = None):
		sns.pairplot(self.df, vars = vars, hue = hue)
		#plt.show()