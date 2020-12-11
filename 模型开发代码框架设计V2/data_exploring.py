'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-19 14:05:07
LastEditors: Troy Wu
LastEditTime: 2020-12-11 16:10:52
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

class Explore:
    def __init__(self, df):
        self.df = df

    def describe_num(self):
        df_num = self.df.select_dtypes(exclude = [np.object])
        des_num = df_num.describe(percentiles = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.9, 0.95, 0.99]).T.\
        assign(**{'偏度': np.array([df_num[col].skew() for col in df_num.columns]),
            '峰度': np.array([df_num[col].kurt() for col in df_num.columns]),
            'na_counts': np.array([df_num[col].isnull().sum() for col in df_num.columns]),
            'na_pct': np.array([df_num[col].isnull().sum() / df_num.shape[0] for col in df_num.columns])}).sort_index(axis = 1)
        return des_num[['min', '12.5%', '25%', '37.5%', '50%', '62.5%', '75%', '87.5%', '90%', '95%', '99%', 'max', \
            'count', '偏度', '峰度', 'mean', 'std', 'na_pct', 'na_counts']]
    
    def describe_obj(self):
        df_obj = self.df.select_dtypes(include = [np.object])
        des_obj = df_obj.describe().T.assign(
            **{'na_counts': np.array([self.df[col].isnull().sum() for col in df_obj.columns]),
            'na_pct': np.array([self.df[col].isnull().sum() / df_obj.shape[0] for col in df_obj.columns])}).sort_index(axis = 1)
        return des_obj

    def corr_and_plot(self, figsize = (32, 18)):
        plt.subplots(figsize = figsize)
        sns.heatmap(self.df.corr().round(2), annot = True, cmap='RdBu')
        plt.show()

    def distplot(self):
        df_num = self.df.select_dtypes(exclude = [np.object]).dropna()
        f, ax = plt.subplots(len(df_num.columns), 1, figsize = (10, len(df_num.columns) * 6))
        for n, col in enumerate(df_num.columns):
            sns.distplot(df_num[col], ax = ax[n])
            plt.title(col)
        plt.show()

    def pairplot(self, vars = None, hue = None):
        sns.pairplot(self.df, vars = vars, hue = hue)
        plt.show()

    def plot_distplot_and_probplot(self):
        df_num = self.df.select_dtypes(exclude = [np.object]).dropna()
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
        
    def VIF(self):
        df_num = self.df.select_dtypes(exclude = [np.object])
        min_max_scaler = MinMaxScaler().fit(df_num)
        data_scaler = pd.DataFrame(min_max_scaler.transform(df_num), columns = df.columns)
        X = np.matrix(data_scaler)
        X = add_constant(X, prepend = False)
        return dict(zip(list(data_scaler.columns)), [variance_inflation_factor(X, i) for i in range(X.shape[1])])
        
            
def compare_train_test(tarin_data, test_data):
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