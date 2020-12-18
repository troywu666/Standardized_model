'''
@Description: 
@Version: 3.0
@Autor: Troy Wu
@Date: 2020-02-12 16:44:46
LastEditors: Troy Wu
LastEditTime: 2020-12-18 18:04:06
'''

# 首次运行可能需要将那些包所在路径加入到环境变量当中
import sys
sys.path.append(r'D:\troywu666\business_stuff\民生对公项目\模型标准化')
from evaluation import Metrics, Metrics_comparison
from train_val import Model_training, bay_opt_lgb
from model_io import Model_pickle
from data_exploring import Explore
from feature_selection import Selector, iv_filter
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

# 载入数据集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df.iloc[2, -1] = np.nan

# 实例化日志打印模块
path = r'D:/troywu666/business_stuff/民生对公项目/模型标准化'
save_log = Save_log(path, 'test')

path = r'D:/troywu666/business_stuff/民生对公项目/模型标准化'
save_log.logging('DataFrame load complete.')

'''lr = LogisticRegression()
lr.fit(data.data, data.target)
y_pred = lr.predict_proba(data.data)[:, 1]
y_pred = {'rf': y_pred, "lr": y_pred}
'''
# 数据探索
num_des = Explore(df).describe_num()
Explore(df).corr_and_plot()
Explore(df).distplot()
#Explore(df).pairplot()
save_log.logging('Exploring complete.')


# 数据预处理
#transformer, transformed_data = Transformer(df['mean radius']).box_cox()
transformer_1, transformed_data = Transformer(df).scaler('standard')
transformer, transformed_data = Transformer(transformed_data).fillna(fill_value = 2, strategy = 'constant')
#transformer, transformed_data = Transformer(transformed_data).encoder(method = 'binarizer', threshold = 3)
save_log.logging('Data preprocessing complete.')

# 特征选择
selector, selected_data = Selector(transformed_data).filter(data.target, k = 30, method = 'mutual_clas_filter')
selector, selected_data = Selector(transformed_data).wrapper(RandomForestClassifier(), data.target, n = 30, step = 10)
selector, selected_data = Selector(transformed_data).embedded(LogisticRegression(), data.target, threshold = 0.03)
selector, selected_data = Selector(transformed_data).variance(threshold = 0.001)
selector, selected_data = Selector(df).filter(data.target, k = 30, method = 'IV')

iv_filter(df, data.target)
save_log.logging('Feature selection complete.')

# 模型训练
model_xgb, y_true, y_pred_xgb = Model_training(selected_data, data.target, test_size = 0.3).xgb_model()
model_lgb, y_true, y_pred_lgb = Model_training(selected_data, data.target, test_size = 0.3).lgb_model()
save_log.logging('Model training complete.')

bay = bay_opt_lgb(df, data.target)

# 模型保存
Model_pickle().dump(transformer, path, 'transformer')
Model_pickle().dump(selector, path, 'selector')
Model_pickle().dump(model_xgb, path, 'xgb_model')
Model_pickle().dump(model_lgb, path, 'lgb_model')
save_log.logging('Model dump complete.')

# 模型加载
transformer = Model_pickle().load(path, 'transformer')
selector = Model_pickle().load(path, 'selector')
model_xgb = Model_pickle().load(path, 'xgb_model')
model_lgb = Model_pickle().load(path, 'lgb_model')
save_log.logging('Model load complete.')

# 模型评估
pred = {'xgb': y_pred_xgb, 'lgb': y_pred_lgb}
compare_score = Metrics_comparison(y_true, pred).compare_score()
Metrics_comparison(y_true, pred).compare_plot()
save_log.logging('Evaluation complete.')

# 模型预测
transformed_data_1 = Prediction().data_preprocess(transformer_1, df)
transformed_data_2 = Prediction().data_preprocess(transformer, df)
selected_data = Prediction().feature_select(selector, transformed_data_2)
predictions = Prediction().model_predict(model_lgb, df, model_cate = 'lgb')
save_log.logging('Prediction complete.')