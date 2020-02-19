'''
@Description: 
@Version: 3.0
@Autor: Troy Wu
@Date: 2020-02-12 16:44:46
@LastEditors: Troy Wu
@LastEditTime: 2020-02-19 16:16:20
'''

#import os
#print(os.getcwd())
import sys
sys.path.append(r'D:\troywu666\business_stuff\民生对公项目\模型标准化')
import seaborn as sns
from evaluation import Metrics, Metrics_comparison
from train_val import Model_training
from model_io import Model_pickle
from data_exploring import Explore
from feature_selection import Selector
from preprocessing import Transformer
from predict import Prediction
from print_in_log import Save_log
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#import logging
#import configparser

'''config = configparser.ConfigParser()
config['server'] = {'user': 'dbc', 'password': 'pass', 'host': '192.168.253.231', 'port': '22', 'dbname': 'server1'}
with open(r'D:/troywu666/business_stuff/民生对公项目/模型标准化/configuration.ini', 'w') as f:
    config.write(f)'''

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['none'] = np.random.randn(df.shape[0])
df['none'] = df['none'].astype('str')
df.iloc[2, -1] = np.nan

path = r'D:/troywu666/business_stuff/民生对公项目/模型标准化'
save_log = Save_log(path, 'test')

path = r'D:/troywu666/business_stuff/民生对公项目/模型标准化'
save_log.logging('DataFrame load complete.')

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

save_log.logging('Exploring complete.')

transformer, df = Transformer(df).scaler('standard')
transformer, df = Transformer(df).fillna(fill_value = 2, strategy = 'constant')
transformer, df = Transformer(df).encoder(method = 'binarizer', threshold = 3)

save_log.logging('Data preprocessing complete.')

selector, transformed_data = Selector(df).variance(threshold = 0.001)
selector, transformed_data = Selector(df).filter(data.target, k = 30, method = 'mutual_clas_filter')
selector, transformed_data = Selector(df).wrapper(RandomForestClassifier(), data.target, n = 30, step = 10)
selector, transformed_data = Selector(df).embedded(RandomForestClassifier(), data.target, threshold = 0.03)

save_log.logging('Feature selection complete.')

model_xgb, y_true, y_pred_xgb = Model_training(data.data, data.target, test_size = 0.3).xgb_model()
model_lgb, y_true, y_pred_lgb = Model_training(data.data, data.target, test_size = 0.3).lgb_model()
pred = {'xgb': y_pred_xgb, 'lgb': y_pred_lgb}
Metrics_comparison(y_true, pred).compare_score()
Metrics_comparison(y_true, pred).compare_plot()

save_log.logging('Model training complete.')

Model_pickle().dump(selector, path, 'selector')
Model_pickle().dump(model_xgb, path, 'example')
selector = Model_pickle().load(path, 'selector')
save_log.logging('Model pickle complete.')

Prediction().feature_select(selector, df)
Prediction().model_predict(model_lgb, df, model_cate = 'lgb')

save_log.logging('Predict complete.')

np.array([[1,2,3], [2,4,5]]).ndim