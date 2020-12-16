# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 13:15:32
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-12-14 21:26:49

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from bayes_opt import BayesianOptimization

class Model_training():
	def __init__(self, train, target, test_size = 0.3):
		self.train = train
		self.target = target
		self.test_size = test_size

	def xgb_model(self, num_boost_round = 1000, early_stopping_rounds = 10, \
					params = {
							'booster': 'gbtree',
							'objective': 'binary:logistic',
							'mat_depth': 6,
							'subsample': 0.7,
							'colsample_bytree': 0.7,
							'eta': 0.1,
							'lambda': 0.1,
							'eval_metric': 'auc',
							'scale_pos_weight': 0.1
							}):
		X_train, X_test, y_train, y_test = train_test_split(self.train, self.target, test_size = self.test_size, random_state = 2020)

		train_data = xgb.DMatrix(X_train, label = y_train)
		val_data = xgb.DMatrix(X_test, label = y_test)

		watch_list = [(train_data, 'train'), (val_data, 'eval')]
		bst = xgb.train(params = params, dtrain = train_data, evals = watch_list, \
			num_boost_round = num_boost_round, early_stopping_rounds = early_stopping_rounds)
		y_pred = bst.predict(val_data)
		return bst, y_test, y_pred

	def lgb_model(self, num_boost_round = 1000, early_stopping_rounds = 10, categorical_feature = 'auto', \
		 			params = {
							'boosting': 'gbdt',
							'objective': 'binary',
							'max_depth': 6,
							'feature_fraction': 0.7,
							#'is_unbalance': True,
							'learning_rate': 0.1,
							'lambda_l1': 0.1,
							'metric': {'auc', 'binary_logloss'},
							'scale_pos_weight': 0.1
							}):
		X_train, X_test, y_train, y_test = train_test_split(self.train, self.target, test_size = self.test_size, random_state = 2020)
		

		train_data = lgb.Dataset(X_train, label = y_train)
		val_data = lgb.Dataset(X_test, label = y_test, reference = train_data)

		
		bst = lgb.train(params = params, train_set = train_data, num_boost_round = num_boost_round, \
			valid_sets = [train_data, val_data], early_stopping_rounds = early_stopping_rounds, categorical_feature = categorical_feature)
		y_pred = bst.predict(X_test)
		return bst, y_test, y_pred

class SBBTree:
    def __init__(self, params, stacking_num, bagging_num, bagging_test_size,
                 num_boost_round, early_stopping_rounds,kf_label = True, categorical_feature = 'auto'):
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []
        self.kf_label = kf_label
        
    def fit(self, X, y):
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits = self.stacking_num, shuffle = True, random_state = 1)
            for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]
                
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test)
                gbm = lgb.train(self.params,
                                lgb_train,
                                num_boost_round = self.num_boost_round,
                                valid_sets = lgb_eval,
                                early_stopping_rounds = self.early_stopping_rounds)
                self.stacking_model.append(gbm)
                pred_y = gbm.predict(X_test, num_iteration = gbm.best_iteration)
                layer_train[test_index, 1] = pred_y
            X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        
        if not self.kf_label:
            for bn in range(self.bagging_num):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.bagging_test_size,
                                                                    random_state = bn)
                lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = self.categorical_feature)
                lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train)
                gbm = lgb.train(self.params,
                                lgb_train, 
                                num_boost_round = 10000,
                                valid_sets = [lgb_train, lgb_eval],
                                early_stopping_rounds = 200)
                self.bagging_model.append(gbm) 
                
        elif self.kf_label:  
            for n, (train_index, test_index) in enumerate(StratifiedKFold(self.bagging_num, shuffle = True, random_state = 0).split(X, y)):   
                X_train = X.iloc[train_index, :]
                y_train = y[train_index]
                X_test = X.iloc[test_index, :]
                y_test = y[test_index]  
                lgb_train =  = lgb.Dataset(X_train, y_train, categorical_feature = self.categorical_feature)
                lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train)
                gbm = lgb.train(self.params,
                                lgb_train, 
                                num_boost_round = 10000,
                                valid_sets = [lgb_train, lgb_eval],
                                early_stopping_rounds = 200)
                self.bagging_model.append(gbm) 
            
    def predict(self, X_pred):
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                pred = gbm.predict(X_pred, num_iteration = gbm.best_iteration)
                test_pred[:, sn] = pred
            X_pred = np.hstack((X_pred, test_pred.mean(axis = 1).reshape(-1, 1)))
        else:
            pass

        for bn, gbm in enumerate(self.bagging_model):
            pred = gbm.predict(X_pred, num_iteration = gbm.best_iteration)
            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred
        return pred_out / self.bagging_num

def bay_opt(X, y):
    d_train = lgb.Dataset(df, data.target, free_raw_data=False)
    def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
        params = {"metric" : 'auc',
                        'max_depth': int(max(max_depth, 1)),
                        'learning_rate': np.clip(0, 1, learning_rate),
                        'num_leaves': int(max(num_leaves, 1)),
                        'n_estimators': int(max(n_estimators, 1))}
        cv_result = lgb.cv(params, d_train, nfold = 5, seed = 0, verbose_eval = 200, stratified = False)
        return 1.0 * np.array(cv_result['auc-mean']).max()

    lgbBO = BayesianOptimization(lgb_eval, {'max_depth': (4, 8),
                                                'learning_rate': (0.05, 0.2),
                                                'num_leaves' : (20,1500),
                                                'n_estimators': (5, 200)}, random_state=0)

    lgbBO.maximize(init_points=5, n_iter=50,acq='ei')
    return lgbBO.max