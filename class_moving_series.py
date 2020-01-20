#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:11:50 2019

@author: jiaqijiang
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

#%%
class moving_series():
    
    def __init__(self, dataframe, columns, windows, y_col, alpha=[0.9], beta=[0.9],corr_coef=0.50, min_periods=1, win_type=None):
        self.df = dataframe
        self.columns = columns
        self.windows = windows
        self.min_periods = min_periods
        self.alpha = alpha
        self.beta = beta
        self.corr_coef = corr_coef
        self.win_type = win_type
        self.y_col = y_col
    

    def create_rcorr_feats(self):
        
        target_col = self.y_col
        corr_matrix = pd.DataFrame(self.df.corr()[target_col])
        corr_columns = list(corr_matrix[corr_matrix[target_col] >= self.corr_coef].index)
        print('Columns used for calculating moving corr are', corr_columns)
    
        for w in self.windows:
            for column in corr_columns:
                self.df['_'.join([target_col, 'rcorr', column, str(w)])] = \
                self.df[target_col].rolling(window=w,min_periods=self.min_periods).corr(self.df[column]).values 
        return self.df

    
    def create_rmean_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rmean', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).mean().values
        return self.df

    def create_rmin_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rmin', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).min().values
        return self.df

    def create_rmax_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rmax', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).max().values
        return self.df

    def create_rmedian_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rmedian', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).median().values
        return self.df
    
    def create_rvar_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rvar', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).var().values
        return self.df

    def create_rskew_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rskew', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).skew().values
        return self.df
    
    def create_rkurt_feats(self):
        for target_col in self.columns:
            for w in self.windows:
                self.df['_'.join([target_col, 'rkurt', str(w)])] = \
                    self.df[target_col].rolling(window=w, min_periods=self.min_periods).kurt().values
        return self.df
    
    def create_ewm_mean_feats(self):
        for target_col in self.columns:
            for a in self.alpha:
                self.df['_'.join([target_col,'ewm_mean', str(a)])] = \
                self.df[target_col].ewm(alpha=a).mean().values
        return self.df
    
    def create_ewm_var_feats(self):
        for target_col in self.columns:
            for a in self.alpha:
                self.df['_'.join([target_col,'ewm_var', str(a)])] = \
                self.df[target_col].ewm(alpha=a).var().values
        return self.df
    
    def double_exponential_smoothing(self,series,a,b):

    # first value is same as series
        alpha = a
        beta = b
        result = []
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # forecasting
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)
        return np.array(result)
    
    def create_double_ewm_feats(self):
        for target_col in self.columns:
            new = self.df[target_col].values
            for a in self.alpha:
                for b in self.beta:
                    self.df['_'.join([target_col,'double_ewm', 'a',str(a), 'b',str(b)])] = \
                        self.double_exponential_smoothing(new,a,b)

        return self.df
 
    
    def select_create(self, select_list):
        if 'corr' in select_list:
            self.create_rcorr_feats()
        if 'mean' in select_list:
            self.create_rmean_feats()
        if 'min' in select_list:
            self.create_rmin_feats()
        if 'max' in select_list:
            self.create_rmax_feats()
        if 'median' in select_list:
            self.create_rmedian_feats()
        if 'var' in select_list:
            self.create_rvar_feats()
        if 'skew' in select_list:
            self.create_rskew_feats()
        if 'kurt' in select_list:
            self.create_rkurt_feats()
        if 'ewm_mean' in select_list:
            self.create_ewm_mean_feats()
        if 'ewm_var' in select_list:
            self.create_ewm_var_feats()
        if 'double_ewm' in select_list:
            self.create_double_ewm_feats()
        return self.df



#%%
dd_MWS = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\DrHartmann project\Dr_Hartmann_Project\52692_MWS_output.csv').iloc[:,:27]
#%%
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

#%%
#Select some columns with variance > 0.09
def select_feat_cols(data,y_col):
    list1 = []
    X = data[data.columns.difference([y_col])].iloc[:,1:]
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    X_1 = abs(sel.fit_transform(X))
    list1 = sel.get_support(indices=True).tolist()
    df1_X = pd.DataFrame(X_1,columns=[X.columns[i] for i in list1])  
    
    return df1_X.columns.tolist()
#%%
columns_var = select_feat_cols(dd_MWS,'KK1_ABS1_PH1')
print('Columns with variance > 0.09 are',columns_var)

#%%
df = dd_MWS.copy()

ms = moving_series(df,columns_var,windows = [i for i in range(20,1441,50)],y_col='KK1_ABS1_PH1')
select_list0 = ['corr','mean','min','max','medium','var','ewm_mean','double_ewm']
#select_list1 = ['skew']
df_moved = ms.select_create(select_list0)

#%%
print(df_moved.info)
#%%
print(df_moved.shape)
#%%
print(df_moved.columns)
#%%
#replace inf with nas and drop nas, see number of rows left
df_moved_u = df_moved.replace([np.inf, -np.inf], np.nan).dropna()
print(df_moved_u.shape)   








                   
                   
                   
                   
                   
                   
                   
                   

