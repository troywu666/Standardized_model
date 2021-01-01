'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-19 14:05:07
LastEditors: Troy Wu
LastEditTime: 2021-01-01 16:07:52
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import math
from sklearn.ensemble import IsolationForest

class Explore:
    def __init__(self, df):
        self.df = df

    def describe_num(self):
        df_num = self.df.select_dtypes(include = [np.number])
        des_num = df_num.describe(percentiles = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.9, 0.95, 0.99]).T.\
        assign(**{'偏度': np.array([df_num[col].skew() for col in df_num.columns]),
            '峰度': np.array([df_num[col].kurt() for col in df_num.columns]),
            '唯一值个数': np.array([df_num[col].nunique() for col in df_num.columns]),
            'na_counts': np.array([df_num[col].isnull().sum() for col in df_num.columns]),
            'na_pct': np.array([df_num[col].isnull().sum() / df_num.shape[0] for col in df_num.columns])}).sort_index(axis = 1)
        return des_num[['min', '12.5%', '25%', '37.5%', '50%', '62.5%', '75%', '87.5%', '90%', '95%', '99%', 'max', \
            'count', '偏度', '峰度', '唯一值个数', 'mean', 'std', 'na_pct', 'na_counts']]
    
    def describe_obj(self):
        df_obj = self.df.select_dtypes(include = [np.object])
        des_obj = df_obj.describe().T.assign(
            **{'na_counts': np.array([self.df[col].isnull().sum() for col in df_obj.columns]),
            'na_pct': np.array([self.df[col].isnull().sum() / df_obj.shape[0] for col in df_obj.columns])}).sort_index(axis = 1)
        return des_obj
    
    def fill_na_inf_num(self, na_fill = 0, inf_fill = 0):
        df_num = self.df.select_dtypes(exclude = [np.number])
        val = df_num.values
        val[np.isinf(val)] = inf_fill
        val[np.isnan(val)] = na_fill
        return pd.DataFrame(val, columns = df_num.columns)

    def corr_and_plot(self): 
        mcorr = self.df.corr()
        mask = np.zeros_like(mcorr, dtype=np.bool) 
        mask[np.triu_indices_from(mask)] = True
        plt.subplots(figsize = (5*len(mcorr), 5*len(mcorr)))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(mcorr, mask = mask, annot = True, cmap = cmap, fmt='0.2f')
        plt.show()

    def distplot(self):
        df_num = self.df.select_dtypes(include = [np.number]).dropna()
        f, ax = plt.subplots(len(df_num.columns), 1, figsize = (10, len(df_num.columns) * 6))
        for n, col in enumerate(df_num.columns):
            sns.distplot(df_num[col], ax = ax[n], kde_kws = {'bw': 1.5})
            plt.title(col)
        plt.show()

    def pairplot(self, vars = None, hue = None):
        sns.pairplot(self.df, vars = vars, hue = hue)
        plt.show()
        
    def boxenplot(self):
        df_num = self.df.select_dtypes(include = [np.number]).dropna()
        #f, ax = plt.subplots(math.ceil(len(df_num.columns)/3), 3)  # 指定绘图对象宽度和高度
        for i, col in enumerate(df_num.columns):
            #a, b = divmod(i, 3)
            plt.subplot(math.ceil(len(df_num.columns)/3), 3, i + 1)  # 15行3列子图
            sns.boxenplot(df_num[col], orient = "v", width = 0.5)  # 箱式图
            plt.ylabel(col, fontsize = 8)
        plt.show()

    def plot_distplot_and_probplot(self):
        try:
            df_num = self.df.select_dtypes(include = [np.number]).dropna()
            train_cols = 6
            train_rows = len(df_num.columns)
            plt.figure(figsize = (5*train_cols, 5*train_rows))

            i = 0
            for col in df_num.columns:
                i += 1
                ax = plt.subplot(train_rows, train_cols, i)
                sns.distplot(df_num[col], fit = stats.norm)
                
                i += 1
                ax = plt.subplot(train_rows, train_cols, i)
                res = stats.probplot(df_num[col], plot = plt)
            plt.show()
        except:
            print("矩阵不存在逆矩阵")

    def anomaly_detection(self, params = {'n_estimators': 100}):
        df_num = self.df.select_dtypes(include = [np.number])
        iso = IsolationForest(**params)
        return iso, iso.fit_predict(df_num)
        
    def VIF(self):
        df_num = self.df.select_dtypes(include = [np.number])
        min_max_scaler = MinMaxScaler().fit(df_num)
        data_scaler = pd.DataFrame(min_max_scaler.transform(df_num), columns = df_num.columns).dropna()
        X = np.matrix(data_scaler)
        X = add_constant(X, prepend = False)
        return dict(zip(list(data_scaler.columns), [variance_inflation_factor(X, i) for i in range(X.shape[1])]))
        
            
def compare_train_test(train_data, test_data):
    dist_cols = 6
    dist_rows = len(test_data.columns)
    plt.figure(figsize = (5*dist_cols, 5*dist_rows))

    for i, col in enumerate(test_data.columns):
        ax = plt.subplot(dist_rows, dist_cols, i+1)
        ax = sns.kdeplot(train_data[col], color = 'Red', shade = True)
        ax = sns.kdeplot(test_data[col], color = 'Blue', shade = True)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend(['train', 'test'])
    plt.show()
    
def cate_countplot(cate_col, label, data):
    plt.figure(figsize = (8, 8))
    plt.title('{} Vs {}'.format(cate_col, label))
    ax = sns.countplot(cate_col, data = data.fillna('missing'), hue = label)
    for p in ax.patches:
        height = p.get_height() 
        plt.text(p.get_x(), height+500, height)
    plt.show()
    
def cate_distribution(label, data):
    plt.subplot(1, 2, 1)
    ax = sns.countplot(x = label, data = data)
    for p in ax.patches:
        height = p.get_height() 
        plt.text(p.get_x(), height+500, height)
    
    plt.subplot(1, 2, 2)
    label_counts = data[label].value_counts().sort_index()
    label_counts.plot(kind = 'pie')
    plt.show()
    return label_counts

def num_label_distplot(num_cols, label, data):
    
    dist_rows = len(num_cols)
    dist_cols = len(label)
    f, ax = plt.subplots(dist_rows, dist_cols, figsize = (5*dist_cols, 5*dist_rows))

    i = 0
    for col in num_cols:
        for label_ in list(data[label].unique()):
            axes = sns.distplot(data[col][data[label] == label_], ax = ax[i])
            axes.set_xlabel(col)
            axes.set_ylabel('Distribution')
    plt.show()