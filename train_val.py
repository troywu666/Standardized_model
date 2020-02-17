# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 13:15:32
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-17 00:45:56

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

class model_training():
	def __init__(self, train, target, test_size):
		self.train = train
		self.target = target
		self.test_size = test_size

	def xgb_model(self):
		X_train, X_test, y_train, y_test = train_test_split(self.train, self.target, test_size = self.test_size)
		train_data = xgb.DMatrix(X_train, label = y_train)
		val_data = xgb.DMatrix(X_test, label = y_test)

		params = {
		'booster': 'gbtree',
		'objective': 'binary:logistic',
		'mat_depth': 3,
		'subsample': 0.7,
		'colsample_bytree': 0.7,
		'eta': 0.1,
		'lambda': 0.1,
		'eval_metric': 'auc',
		'scale_pos_weight': 0.1
		}
		watch_list = [(train_data, 'train'), (val_data, 'eval')]
		bst = xgb.train(params, train_data, evals = watch_list, num_boost_round = 1000, early_stopping_rounds = 10)
		y_pred = bst.predict(val_data)
		return bst, y_test, y_pred

	def lgb_model(self):
		X_train, X_test, y_train, y_test = train_test_split(self.train, self.target, test_size = self.test_size)
		train_data = lgb.Dataset(X_train, label = y_train)
		val_data = lgb.Dataset(X_test, label = y_test, reference = train_data)

		params = {
		'boosting': 'gbdt',
		'objective': 'binary',
		'mat_depth': 3,
		'feature_fraction': 0.7,
		#'is_unbalance': True,
		'eta': 0.1,
		'lambda_l1': 0.1,
		'metric': 'auc',
		'scale_pos_weight': 0.1
		}
		bst = lgb.train(params, train_data, num_boost_round = 1000, valid_sets = [train_data, val_data], early_stopping_rounds = 10)
		y_pred = bst.predict(X_test)
		return bst, y_test, y_pred