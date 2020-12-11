# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-14 13:25:27
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-04-22 12:06:38

import xgboost as xgb
import lightgbm as lgb

class Prediction():
	def __init__(self):
		pass

	def feature_select(self, selector, test):
		return selector.transform(test)
		
	def data_preprocess(self, transformer, test):
		return transformer.transform(test)

	def model_predict(self, model, test, model_cate = 'xgb', pred_leaf = False):
		if model_cate == 'xgb':
			test_data = xgb.DMatrix(test)
		if model_cate == 'lgb':
			test_data = test
		return model.predict(test_data, pred_leaf = pred_leaf)