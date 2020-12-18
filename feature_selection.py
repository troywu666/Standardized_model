'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-11 04:34:22
LastEditors: Troy Wu
LastEditTime: 2020-12-18 18:05:00
'''

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
		selector = VarianceThreshold(threshold).fit(self.fea)
		return selector, selector.transform(self.fea)

	def filter(self, y, k, method = 'chi2_filter'):
		'''
		method可选“var_filter”、“chi2_filter”、“f_clas_filter、“f_reg_filter”、“mutual_clas_filter”、“mutual_reg_filter”、“IV”七种
		'''
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
		elif method == 'IV':
			def cal_iv(x, y, n_bins=10, null_value=np.nan,):
				# 剔除空值
				x = x[x != null_value]
				
				# 若 x 只有一个值，返回 0
				if len(x.unique()) == 1 or len(x) != len(y):
					return 0
				
				if x.dtype == np.number:
					# 数值型变量
					if x.nunique() > n_bins:
						# 若 nunique 大于箱数，进行分箱
						x = pd.qcut(x, q=n_bins, duplicates='drop')
							
				# 计算IV
				groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
				t0, t1 = pd.Series(y).value_counts().index
				groups = groups / groups.sum()
				not_zero_index = (groups[t0] > 0) & (groups[t1] > 0)
				groups['iv_i'] = (groups[t0] - groups[t1]) * np.log(groups[t0] / groups[t1])
				groups['iv_i'][~not_zero_index] = 0 
        		iv = groups['iv_i'].sum(axis = 0)
			
				return iv

			def IV(data, label):
				data = pd.DataFrame(data)
				return data.apply(lambda x: cal_iv(x, label), axis=0)
			selector = SelectKBest(IV, k = k).fit(self.fea, y)
			return selector, selector.transform(self.fea)
		else:
			print('The parameters were worng!')

	def wrapper(self, estimator, y, n, step = None):
		selector = RFE(estimator, n, step).fit(self.fea, y)
		return selector, selector.transform(self.fea)
	
	def embedded(self, estimator, y, threshold):
		selector = SelectFromModel(estimator, threshold).fit(self.fea, y)
		return selector, selector.transform(self.fea)

def iv_filter(data, label):
    def cal_iv(x, y, n_bins=10, null_value=np.nan,):
        #y = pd.DataFrame(y)
        # 剔除空值
        x = x[x != null_value]
        
        # 若 x 只有一个值，返回 0
        if len(x.unique()) == 1 or len(x) != len(y):
            return 0
        
        if x.dtype == np.number:
            # 数值型变量
            if x.nunique() > n_bins:
                # 若 nunique 大于箱数，进行分箱
                x = pd.qcut(x, q=n_bins, duplicates='drop')
                    
        # 计算IV
        groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
        t0, t1 = pd.Series(y).value_counts().index
        groups = groups / groups.sum()
        not_zero_index = (groups[t0] > 0) & (groups[t1] > 0)
        groups['iv_i'] = (groups[t0] - groups[t1]) * np.log((groups[t0]) / (groups[t1]))
        groups['iv_i'][~not_zero_index] = 0 
        iv = groups['iv_i'].sum(axis = 0)
        return iv
    return data.apply(lambda x: cal_iv(x, label), axis=0)
