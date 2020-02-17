'''
@Description: 
@Version: 3.0
@Autor: Troy Wu
@Date: 2020-02-12 16:44:46
@LastEditors: Troy Wu
@LastEditTime: 2020-02-17 18:46:08
'''

import logging
#import os
#print(os.getcwd())
import sys
sys.path.append(r'D:\troywu666\business_stuff\民生对公项目\模型标准化')
from evaluation import metrics, metrics_comparison
from train_val import model_training
from model_io import model_pickle
from data_exploring import Explore
from feature_selection import Selector
from preprocessing import Transformer
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import configparser

config = configparser.ConfigParser()
config['server'] = {'user': 'dbc', 'password': 'pass', 'host': '192.168.253.231', 'port': '22', 'dbname': 'server1'}
with open(r'D:/troywu666/business_stuff/民生对公项目/模型标准化/configuration.ini', 'w') as f:
    config.write(f)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['none'] = np.random.randn(len(df['none']))
df['none'] = df['none'].astype('str')
df.iloc[2, -1] = np.nan

'''lr = LogisticRegression()
lr.fit(data.data, data.target)
y_pred = lr.predict_proba(data.data)[:, 1]
y_pred = {'rf': y_pred, "lr": y_pred}
'''
Explore(df).describe_num()
Explore(df).describe_obj()
Explore(df).corr_and_plot()
Explore(df).distplot()
#Explore(df).pairplot()

#select(df).variance()
#select(df).filter(data.target, k = 30, method = 'mutual_clas_filter')
#select(df).with_model(RandomForestClassifier(), data.target, threshold = 30, step = 10, method = 'wrapper')
#select(df).with_model(RandomForestClassifier(), data.target, threshold = 0.03, method = 'embedded')

Transformer(df).scaler('standard')
Transformer(df).fillna(fill_value = 2, strategy = 'constant')
Transformer(df.iloc[:, : -1]).encoder(method = 'binarizer', threshold = 3)

model_xgb, y_true, y_pred_xgb = model_training(data.data, data.target, test_size = 0.3).xgb_model()
model_lgb, y_true, y_pred_lgb = model_training(data.data, data.target, test_size = 0.3).lgb_model()
pred = {'xgb': y_pred_xgb, 'lgb': y_pred_lgb}
metrics_comparison(y_true, pred).compare_score()
metrics_comparison(y_true, pred).compare_plot()

path = r'D:/troywu666/business_stuff/民生对公项目/模型标准化'
model_pickle(path, 'example').dump(model_xgb)
model_pickle(path, 'example').load_predict(data.data, 'xgb')