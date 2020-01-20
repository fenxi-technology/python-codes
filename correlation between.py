# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:40:27 2020

@author: ThinkPad
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
#%%
#target_col is from df2
def check_correlation_between(df1,df2,target_col,coff):
    
    df = df1.merge(df2[['Time',target_col]],on='Time')
    df_corr = df.iloc[:,1:].corr()[target_col]
    feature_list = df_corr[abs(df_corr)>coff].sort_values(ascending=False)
    
    return feature_list


#%%
def main_correlation_between(name,target_col,coff):
    
    df = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\code\shaoxingdianlishebei\initial\%s.csv'%name).iloc[:,1:]
    obj = ['Time','OrderID','WorkID']
    ex = explorative_analysis(df,obj,None,None)
    ex.convert_dtype()
    df = ex.check_missing()
    
    om = operate_influxdb('myDB')
    namelist = []
    for i in range(len(list(om.show_all_tables())[0])):
    
        nam = list(om.show_all_tables())[0][i]['name']
        if (nam != name) and (nam != '13中压铜屏电表') and (nam != '14中压铠装电表') and \
        (nam != '15中压铜大拉电表') and (nam != '16中压交联线电表') == True:
            namelist.append(nam)
    print(namelist)  
    
    for i in namelist:
        data = main_corr(i,None,None)
        feature_list = check_correlation_between(data,df,target_col,coff)
        print('There are {} strongly correlated values from %s with %s:\n{}'.format(len(feature_list),feature_list)%(i,target_col))
        

#%%
main_correlation_between('0拉丝机','LSV2',0.1)
#%%




























