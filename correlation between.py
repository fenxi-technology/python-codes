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

def main_corr(name,col,coff):
    #col = 'LSV2' or None
    #df = influxdb_df_to_python_df_update(name)
    df = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\code\shaoxingdianlishebei\initial\%s.csv'%name).iloc[:,1:]
    obj = ['Time','OrderID','WorkID']
    ex = explorative_analysis(df,obj,col,coff)
    ex.convert_dtype()
    result = ex.check_missing()  
    
    return result
#%%
#target_col is from df2
def check_correlation_between(df1,df2,target_col,coff):
    
    df = pd.concat([df1,df2[target_col]],axis=1)
    print('Shape after merging is',df.shape)
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

    
    for i in namelist:
        print(i + '...')
        data = main_corr(i,None,None)
        feature_list = check_correlation_between(data,df,target_col,coff)
        print('There are {} strongly correlated values from %s with %s:\n{}'.format(len(feature_list),feature_list)%(i,target_col))
        

#%%
main_correlation_between('0拉丝机','LSV13',0.5)
#%%
main_correlation_between('1绞线机','JXV12',0.5)
#%%
main_correlation_between('3交联机2','JL2V187',0.5)
#%%
#null
main_correlation_between('9成缆','CLV3',0.5)
#%%
#JSV15:计米显示 JSV29:加装采集长度 JSV33:直径1 JSV34：直径2

main_correlation_between('10挤塑机','JSV15',0.5)
#null
main_correlation_between('10挤塑机','JSV29',0.5)
main_correlation_between('10挤塑机','JSV33',0.5)
main_correlation_between('10挤塑机','JSV34',0.5)
#%%
main_correlation_between('11测径仪','CJV1',0.5)
main_correlation_between('11测径仪','CJV2',0.5)
#%%
main_correlation_between('12测长仪','CCV1',0.5)
main_correlation_between('12测长仪','CCV2',0.5)





