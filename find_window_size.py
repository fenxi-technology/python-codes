# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:45:40 2019

@author: ThinkPad
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

dd_MWS = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\DrHartmann project\Dr_Hartmann_Project\52692_MWS_output.csv').iloc[:,:27]

def format_convert_timestamp(formatTime):
    timeArray = time.strptime(formatTime, "%d.%m.%Y %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp


def timestamp_convert_format(timestamp):
    time_local = time.localtime(timestamp)
    formatTime = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return formatTime


dd_MWS['Datum'] = dd_MWS.iloc[:, 0].apply(lambda x: format_convert_timestamp(x))
dd_MWS['Datum'] = dd_MWS.iloc[:, 0].apply(lambda x: timestamp_convert_format(x))

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

#%%
#Select some columns with variance > 0.16
def select_feat_cols(data,y_col):
    list1 = []
    X = data[data.columns.difference([y_col])].iloc[:,1:]
    sel = VarianceThreshold(threshold=0.4)
    X_1 = abs(sel.fit_transform(X))
    list1 = sel.get_support(indices=True).tolist()
    df1_X = pd.DataFrame(X_1,columns=[X.columns[i] for i in list1])  
    
    return df1_X.columns.tolist()

columns_var = select_feat_cols(dd_MWS,'KK1_ABS1_PH1')
print('Columns with variance > 0.4 are',columns_var)
#%%

windows = [i for i in range(20,1441,50)]
print(len(windows))

# Create Moving_Series variables
ms = moving_series(dd_MWS,columns_var,windows,y_col='KK1_ABS1_PH1')
select_list0 = ['corr','mean','min','max','medium','var','ewm_mean','double_ewm']
#select_list1 = ['skew']
df_moved = ms.select_create(select_list0)
df0 = df_moved.replace([np.inf, -np.inf], np.nan).dropna()

# Create Time Variables
df0['time'] = df0['Datum']
ds = create_time_feature(df0, ['month', 'hourofDay','minuteofHour','dayofWeek','periodOfDay','season', 'isWeekend'])
ds = ds.iloc[:,1:]
ds = ds.drop(labels=['time'], axis=1,inplace = False)
print(ds.shape)

#%%
ds.to_csv(r'C:\Users\ThinkPad\Documents\test sets\code\MWS_fullfeats.csv')
#%%
pd.set_option('display.max_rows',None)
#%%
#Method0: Correlation using select k best
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def feature_selection_correlation(data,y_col,k):
    
    data = abs(data)
    X = data[data.columns.difference([y_col])]
    y = data[y_col]
    ch2 = SelectKBest(chi2, k)
    X_1 = ch2.fit_transform(X, y.astype('int'))
    list_ = ch2.get_support(indices=True).tolist()
    columns = [data.columns.to_list()[i] for i in list_]
    return columns

columns_k = feature_selection_correlation(ds.iloc[:,1:],'KK1_ABS1_PH1',100)
print(columns_k)

#%%
#Method1: Calculate Correlation between Moving_Series Variables
def calcu_corr(data,y_col,k):
    corr_matrix = abs(pd.DataFrame(data.corr()[y_col]))
    corr_matrix_n = corr_matrix.sort_values(by=y_col,ascending=False)[:k]
    return corr_matrix_n

columns_co = calcu_corr(df_moved_u.copy(),'KK1_ABS1_PH1',100)
print(columns_co)
    

#%%
#Method2： Recursive Feature Elimination using Linear Regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression,Ridge,Lasso

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
def rfe_function(data,y_col,k):
    
    X = data[data.columns.difference([y_col])]
    y = data[y_col].astype('float')
    lr = Ridge(alpha=100000, fit_intercept=True, normalize=True, copy_X=True, max_iter=1500, tol=1e-4, solver='auto')
    rfe = RFE(estimator=lr, n_features_to_select=k)
    rfe.fit_transform(X, y)
    ranking = sorted(zip(rfe.ranking_,X.columns.to_list()), reverse=True)[:k]
    
    co_list = []
    for i in range(k):
        co_list.append(ranking[i][1])
    return co_list

columns_rfe = rfe_function(dd_MWS.iloc[:500,1:].dropna(),'KK1_ABS1_PH1',10)
print(columns_rfe)
#%%
print(len(columns_rfe))
print(dd_MWS.iloc[:1000,1:].dropna().shape)
#%%
#Method3: Calculate importance level using Random Forest
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def rf_importance(data,y_col,k):
    
    X = data[data.columns.difference([y_col])]   
    y = data[y_col]
    rf = RandomForestRegressor()
    rf.fit(X, y)
    ranking = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), data.columns.to_list()), reverse=True)[:k]
    print(ranking)
    co_list = []
    for i in range(k):
        co_list.append(ranking[i][1])
    return co_list

columns_rf = rf_importance(ds,'KK1_ABS1_PH1',100)
print(columns_rf)

#%%
def feat_select(data,y_col,k,method):
    
    if method = 'kbest':
        columns = feature_selection_correlation(data.iloc[:,1:],y_col,k)
        selected_data = data[columns]
    if method = 'rfe':
        columns = rfe_function(data.iloc[:,1:],y_col,k)
        selected_data = data[columns]
    if method = 'rf':
        columns = rf_importance(data.iloc[:,1:],y_col,k)
        selected_data = data[columns]
        
    return selected_data


#%%
print(ds.shape)













