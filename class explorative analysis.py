# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:49:16 2020

@author: ThinkPad
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

#%%
class explorative_analysis:
    
    #y_col is a list of columns we would like to explore
    def __init__(self,df,objectlist,y_col,coff):
        self.df = df
        self.objectlist = objectlist
        self.y_col = y_col
        self.coff = coff
        
        
    def get_flo_boo_list(self):
        
        df = self.df
        flo = []
        boo = []
        for i in df.columns.difference(['Time','OrderID','State','WorkID']):
            if df[i].dtype == object:
                boo.append(i)
            elif df[i].dtype == bool:
                boo.append(i)
            elif df[i].dtype == np.float64 or int:
                flo.append(i)
        return flo,boo
      
        
    #convert dtypes, state
    def convert_dtype(self):
        
        flo,boo = self.get_flo_boo_list()
        
        self.df.replace(r'^\s*$', np.nan, regex=True,inplace=True)
        self.df.replace(np.nan, -999, inplace=True)
        self.df[flo] = self.df[flo].astype(np.float64) 
        self.df[boo] = 1*self.df[boo].astype(np.int)
        self.df.replace(-999, np.nan, inplace=True)

        state_ = pd.get_dummies(self.df['State'],prefix='State')
        self.df = pd.concat([self.df[self.objectlist],state_,self.df[flo],self.df[boo]],axis=1)

        return self.df         
 
    
    #check null values and drop/fillin null values
    def check_missing(self):

        #print(self.df.isnull().sum())
        
        #drop_col here are OrderID & WorkID
        drop_col = self.df.loc[:,self.df.isnull().sum()==len(self.df)].columns
        self.df.drop(drop_col,axis=1,inplace=True)
        
        
        flo,boo = self.get_flo_boo_list()
        
        if self.df.isnull().sum().max() > 0.05*len(self.df):
            
            self.df = self.df.dropna()

        else:
            self.df[flo].fillna(lambda x:np.mean(x))
            self.df[boo].fillna(lambda x:np.mode(x))

        print('Shape after dealing with NA values is', self.df.shape)
 
        return self.df
    
    def correlation_matrix(self):
        
        df = self.df.iloc[:,1:].dropna()
        sel = VarianceThreshold(threshold=0)
        X_1 = abs(sel.fit_transform(df))
        list1 = sel.get_support(indices=True).tolist()
        df = pd.DataFrame(X_1,columns=[df.columns[i] for i in list1]) 
        df_corr = df.iloc[:,1:].corr()
        print(df_corr)
        
        
    #select columns with high correlation with y_col
    def check_correlation(self):
        for i in self.y_col:
            df_corr = self.df.iloc[:,1:].corr()[i]
            feature_list = df_corr[abs(df_corr)>self.coff].sort_values(ascending=False)
            print('There are {} strongly correlated values with %s:\n{}'.format(len(feature_list),feature_list)%i)
    
    
    #plot the full correlation matrix
    def plot_df_correlation_full(self):
        df = self.df.iloc[:,1:].dropna()
        f = plt.figure(figsize=(24, 19))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
    
    
    #plot correlation matrix with columns whose variance > 0 
    def plot_df_correlation_use(self):
        df = self.df.iloc[:,1:].dropna()
        sel = VarianceThreshold(threshold=0)
        X_1 = abs(sel.fit_transform(df))
        list1 = sel.get_support(indices=True).tolist()
        df = pd.DataFrame(X_1,columns=[df.columns[i] for i in list1])   
        
        f = plt.figure(figsize=(24, 19))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        
        
    def plot_column_freq(self):
        for i in self.y_col:
            plt.figure(figsize=(19,18))
            sns.distplot(self.df[i].dropna(),color='g', bins=100,hist_kws={'alpha':0.4})
    
    
    def plot_column_dist(self):
        for i in self.y_col:
            plt.figure(figsize=(19,18))
            series = self.df[['Time',i]].set_index('Time')
            plt.plot(series)
        
    def column_describe(self):
        for i in self.y_col:
            print(self.df[i].describe())


#%%
#导出表格成为csv格式
for i in range(len(list(om.show_all_tables())[0])):
    name=list(om.show_all_tables())[0][i]['name']
    influxdb_df_to_python_df_update(name).to_csv(r'C:\Users\ThinkPad\Documents\test sets\code\shaoxingdianlishebei\initial\%s.csv'%name)

#%%

def main(name,col,coff):
    #col = 'LSV2' or None
    #df = influxdb_df_to_python_df_update(name)
    df = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\code\shaoxingdianlishebei\initial\%s.csv'%name).iloc[:,1:]
    obj = ['Time','OrderID','WorkID']
    ex = explorative_analysis(df,obj,col,coff)
    ex.convert_dtype()
    result = ex.check_missing()
    
    #ex.correlation_matrix()
    ex.plot_df_correlation_use()
    
    if col != None:
        ex.column_describe()
        ex.check_correlation()
        #ex.plot_column_dist()
        
    return result


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
#LSV2:长度 LSV11:加装采集长度 LSV12:总长度 LSV13:直径
lsj_ = main('0拉丝机',['LSV2','LSV13'],0.4)
#%%
#JXV1:计米长度 JXV11:加装采集长度 JXV12：直径
jxj_ = main('1绞线机',['JXV1','JXV11','JXV12'],0.4)
#%%
#JL2V187:计米长度
jlj2_ = main('3交联机2',['JL2V187'],0.4)
#%%
#PX1V47：总壁厚偏心值 PX1V49:总壁厚偏心度
px1_ = main('4偏心1',['PX1V47','PX1V49'],0.4)
#%%
#PX2V47：总壁厚偏心值 PX2V49:总壁厚偏心度
px2_ = main('5偏心2',['PX2V47','PX2V49'],0.4)
#%%
#TP1V15:计米长度 TP1V22:加装采集长度
tp1_ = main('6铜屏1',['TP1V15','TP1V22'],0.4)
#%%
#TP2V15:计米长度 TP2V22:加装采集长度
tp2_ = main('7铜屏2',['TP2V15','TP2V22'],0.4)
#%%
#KZV15:计米长度
kz_ = main('8铠装',['KZV15'],0.4)
#%%
#CLV3:计米长度
clj_ = main('9成缆',['CLV3'],0.4)
#%%
#JSV15:计米显示 JSV29:加装采集长度 JSV33:直径1 JSV34：直径2
jsj_ = main('10挤塑机',['JSV15','JSV29','JSV33','JSV34'],0.4)
#%%
#CJV1	拉丝直径 CJV2	绞线直径 CJV3	挤塑机直径1 CJV4	挤塑机直径2
cjy_ = main('11测径仪',['CJV1','CJV2'],0.4)
#%%
#CCV1	拉丝测长 CCV2	绞线测长 CCV3	铜屏1测长 CCV4	铜屏2测长
#CCV5	铠装测长 CCV6	挤塑测长
ccy_ = main('12测长仪',['CCV1','CCV2','CCV3'],0.4)















       
        
        
        
        
        