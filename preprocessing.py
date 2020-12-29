'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-11 06:46:40
LastEditors: Troy Wu
LastEditTime: 2020-12-29 09:59:42
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, Binarizer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm_notebook
import math

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
  
class WoeEncode:
    def __init__(self, data, label, cols):
        data[cols] = data[cols].astype(str)
        self.train = data[data[label].notnull()]
        self.test = data[data[label].isnull()]
        self.label = label
        self.cols = cols
        self.nrows, self.ncols = self.train.shape
        
    def char_woe(self):
        dic = dict(self.train[self.label].value_counts())
        good = dic.get(1, 0) + 1e-10
        bad = dic.get(0, 0) + 1e-10
        for col in self.cols:
            data = dict(self.train.groupby([col, self.label]).size())
            if len(data) > 100:
                print(col, 'contains too many different values.')
                continue
            dic = dict()
            for k, v in data.items():
                value, dp = k
                dic.setdefault(value, {})
                dic[value][int(dp)] = v
            for k, v in dic.items():
                dic[k] = {str(int(k1)): v1 for k1, v1 in v.items()}
                dic[k]['cnt'] = sum(v.values())
                pos_rate = round(v.get('1', 0) / dic[k]['cnt'], 5)
                dic[k]['pos_rate'] = pos_rate
            dic = self.combine_box_char(dic)
            for k, v in dic.items():
                a = v.get('0', 1)/good + 1e-10
                b = v.get('1', 1)/bad + 1e-10
                dic[k]['Pos'] = v.get('1', 0)
                dic[k]['Neg'] = v.get('0', 0)
                dic[k]['woe'] = round(math.log(a/b), 5)
            for klis, v in dic.items():
                for k in klis.split(','):
                    self.train.loc[self.train[col] == k, '{}_woe'.format(col)] = v['woe']
                    if not isinstance(self.test, str):
                        self.test.loc[self.test[col] == k, '{}_woe'.format(col)] = v['woe']
        return pd.concat([self.train, self.test], axis = 0)
    
    def combine_box_char(self, dic):
        while len(dic) >= 10:
            pos_rate_dic = {k: v['pos_rate'] for k,v in dic.items()}
            pos_rate_sorted = sorted(pos_rate_dic.items(), key = lambda x: x[1], reverse = False)
            pos_rate = [pos_rate_sorted[i+1] - pos_rate_sorted[i] for i in range(len(pos_rate_sorted) - 1)]
            min_rate_index = pos_rate.index(min(pos_rate))
            k1, k2 = pos_rate_sorted[min_rate_index][0], pos_rate_sorted[min_rate_index+1][0]
            dic['{},{}'.format(k1, k2)] = dict()
            dic['{},{}'.format(k1, k2)]['1'] = dic[k1].get('1', 0) + dic[k2].get('1', 0)
            dic['{},{}'.format(k1, k2)]['0'] = dic[k1].get('1', 0) + dic[k2].get('0', 0)
            dic['{},{}'.format(k1, k2)]['cnt'] = dic[k1].get('cnt', 0) + dic[k2].get('cnt', 0)
            dic['{},{}'.format(k1, k2)]['pos_rate'] = round(dic['{},{}'.format(k1, k2)]['1']/dic['{},{}'.format(k1, k2)]['cnt'], 5)
            del dic[k1], dic[k2]
            
        min_cnt = min([v['cnt'] for v in dic.values()])
        while min_cnt < self.nrows*0.05 and len(dic) > 5:
            min_key = [k for k, v in dic.items() if v['cnt'] == min_cnt][0]
            pos_rate_dic = {k: v['pos_rate'] for k, v in dic.items()}
            pos_rate_sorted = sorted(pos_rate_dic.items(), key = lambda x: x[1], reverse = False)
            keys = [k[0] for k in pos_rate_sorted]
            min_index = keys.index(min_key)
            if min_index == 0:
                k1, k2 = keys[: 2]
            elif min_index == len(dic)-1:
                k1, k2 = keys[-2: ]
            else:
                before_pos_rate = dic[min_key]['pos_rate'] - dic[keys[min_index-1]]['pos_rate']
                after_pos_rate = dic[keys[min_index+1]]['pos_rate'] - dic[min_key]['pos_rate']
                if before_pos_rate <= after_pos_rate:
                    k1, k2 = keys[min_index-1], min_key
                else:
                    k1, k2 = min_key, keys[min_index+1]
            dic['{},{}'.format(k1, k2)] = dict()
            dic['{},{}'.format(k1, k2)]['1'] = dic[k1].get('1', 0) + dic[k2].get('1', 0)
            dic['{},{}'.format(k1, k2)]['0'] = dic[k1].get('1', 0) + dic[k2].get('0', 0)
            dic['{},{}'.format(k1, k2)]['cnt'] = dic[k1].get('cnt', 0) + dic[k2].get('cnt', 0)
            dic['{},{}'.format(k1, k2)]['pos_rate'] = round(dic['{},{}'.format(k1, k2)]['1']/dic['{},{}'.format(k1, k2)]['cnt'], 5)
            del dic[k1], dic[k2]
            min_cnt = min([v['cnt'] for v in dic.values()])
        return dic