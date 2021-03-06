'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-11 04:34:22
LastEditors: Troy Wu
LastEditTime: 2020-12-21 01:25:13
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

def iv_filter(data, label, n_bins = 10):
    def cal_iv(x, y, n_bins = n_bins, null_value=np.nan):
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
                x = pd.qcut(x, q = n_bins, duplicates='drop')
                    
        # 计算IV
        groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
        t0, t1 = pd.Series(y).value_counts().index
        groups = groups.replace(0, 1) / groups.sum()
        #not_zero_index = (groups[t0] > 0) & (groups[t1] > 0)
        groups['iv_i'] = (groups[t0] - groups[t1]) * np.log((groups[t0]) / (groups[t1]))
        #groups['iv_i'][~not_zero_index] = 0 
        iv = groups['iv_i'].sum(axis = 0)
        return iv
    return data.apply(lambda x: cal_iv(x, label), axis=0)

def get_iv_series(feature, labels, keep_cols=None, cut_bin_dict=None):
    '''
    计算各变量最大的iv值,get_iv_series方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        feature: 数据集的特征空间
        labels: 数据集的输出空间
        keep_cols: 需计算iv值的变量列表
        cut_bin_dict: 数值型变量要进行分箱的阈值字典,格式为{'col1':[value1,value2,...], 'col2':[value1,value2,...], ...}
    ------------------------------------------------------------
    入参结果如下:
        iv_series: 各变量最大的IV值
    '''
    def iv_count(data_bad, data_good):
        '''计算iv值'''
        value_list = set(data_bad.unique()) | set(data_good.unique())
        iv = 0
        len_bad = len(data_bad)
        len_good = len(data_good)
        for value in value_list:
            # 判断是否某类是否为0，避免出现无穷小值和无穷大值
            if sum(data_bad == value) == 0:
                bad_rate = 1 / len_bad
            else:
                bad_rate = sum(data_bad == value) / len_bad
            if sum(data_good == value) == 0:
                good_rate = 1 / len_good
            else:
                good_rate = sum(data_good == value) / len_good
            iv += (good_rate - bad_rate) * math.log(good_rate / bad_rate,2)
        return iv

    if keep_cols is None:
        keep_cols = sorted(list(feature.columns))
    col_types = feature[keep_cols].dtypes
    categorical_feature = list(col_types[col_types == 'object'].index)
    numerical_feature = list(col_types[col_types != 'object'].index)

    iv_series = pd.Series()

    # 遍历数值变量计算iv值
    for col in numerical_feature:
        cut_bin = cut_bin_dict[col]
        # 按照分箱阈值分箱,并将缺失值替换成Blank,区分好坏样本
        data_bad = pd.cut(feature[col], cut_bin, right=False).cat.add_categories(['Blank']).fillna('Blank')[labels == 1]
        data_good = pd.cut(feature[col], cut_bin, right=False
                           ).cat.add_categories(['Blank']).fillna('Blank')[labels == 0]
        iv_series[col] = iv_count(data_bad, data_good)
    # 遍历类别变量计算iv值
    for col in categorical_feature:
        # 将缺失值替换成Blank,区分好坏样本
        data_bad = feature[col].fillna('Blank')[labels == 1]
        data_good = feature[col].fillna('Blank')[labels == 0]
        iv_series[col] = iv_count(data_bad, data_good)

    return iv_series