'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-12-15 18:53:41
LastEditors: Troy Wu
LastEditTime: 2020-12-15 20:05:05
'''
from evaluation import Metrics, Metrics_comparison
from model_io import Model_pickle
from data_exploring import Explore
from feature_selection import Selector
from preprocessing import Transformer
from predict import Prediction
from print_in_log import Save_log
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
#import logging
#import configparser

'''config = configparser.ConfigParser()
config['server'] = {'user': 'dbc', 'password': 'pass', 'host': '192.168.253.231', 'port': '22', 'dbname': 'server1'}
with open(r'D:/troywu666/business_stuff/民生对公项目/模型标准化/configuration.ini', 'w') as f:
    config.write(f)'''

# 载入数据集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)

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
bay_opt(df, data.target)