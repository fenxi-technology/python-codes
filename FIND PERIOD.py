# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:42:17 2019

@author: ThinkPad
"""
#%%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
#%%
dd_MWS = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\DrHartmann project\Dr_Hartmann_Project\52692_MWS_output.csv').iloc[:,:27]

#%%

def format_convert_timestamp(formatTime):
    timeArray = time.strptime(formatTime, "%d.%m.%Y %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp


def timestamp_convert_format(timestamp):
    time_local = time.localtime(timestamp)
    formatTime = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return formatTime


print(format_convert_timestamp('30.06.2015 18:01:18'))
print(timestamp_convert_format(1435658478))

#%%
dd_MWS['time'] =  dd_MWS.iloc[:, 0].apply(lambda x: format_convert_timestamp(x))
dd_MWS['time'] = dd_MWS.iloc[:,-1].apply(lambda x: timestamp_convert_format(x))
dd_MWS['time'] = pd.to_datetime(dd_MWS['time'])
dd_MWS['time'] = dd_MWS['time'].dt.floor('Min')

#%%
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
#%%
date_MWS = dd_MWS.set_index('time')
series = date_MWS['KK1_ABS1_PH1']
print(series)
plt.plot(series)
#%%
def remove_outlier(ts):
    ts = ts.copy()
    dif = ts.diff().dropna()
    td = dif.describe()
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # 定义低点阈值
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])
    forbid_index = dif[(dif > high) | (dif < low)].index
    print(forbid_index)
    print(np.median(ts))
    ts[forbid_index] = np.median(ts)
    plt.plot(ts)
    return ts
#%%
smooth_series = remove_outlier(series)

#%%
def diff_smooth(ts, interval):
    '''时间序列平滑处理'''
    starttime = time.time()

    # 间隔为1小时
    wide = interval/60
    # 差分序列
    dif = ts.diff().dropna()
    # 描述性统计得到：min，25%，50%，75%，max值
    td = dif.describe()
    print(td)
    # 定义高点阈值，1.5倍四分位距之外
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # 定义低点阈值
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])
    print(high)
    print(low)
    i = 0 
    forbid_index = dif[(dif > high) | (dif < low)].index
    print(forbid_index)
    print('异常值数量为',len(forbid_index))
    while i < len(forbid_index) - 1:
        # 发现连续多少个点变化幅度过大
        n = 1 
        # 异常点的起始索引
        start = forbid_index[i]
        if (i+n) < len(forbid_index)-1:
            while forbid_index[i+n] == start + datetime.timedelta(minutes=n):
                n += 1
                #print('n=',n)
        i += n - 1 
        # 异常点的结束索引
        end = forbid_index[i]
        # 用前后值的中间值均匀填充
        before = start - datetime.timedelta(minutes=wide)
        after = end + datetime.timedelta(minutes=wide)
        #跳过一些数据里没有的时间
        if before in ts.index.to_list() and after in ts.index.to_list():
            value = np.linspace(ts[before],ts[after], n)
            ts[start: end] = value
            print(ts[start: end])
        #elif before not in ts.index.to_list() and after in ts.index.to_list():
            #value = np.linspace(ts[start],ts[after], n)
            #ts[start: end] = value
        #elif before in ts.index.to_list() and after not in ts.index.to_list():
            #value = np.linspace(ts[before],ts[end], n)
            #ts[start: end] = value
        #else:
            #value = np.linspace(ts[start],ts[end], n)
            #ts[start: end] = value
        i += 1
        print('i=',i)
        
    endtime = time.time()
    print(endtime-starttime)
    return ts
#%%
ts = diff_smooth(series,120)
#%%
#ts.to_csv(r'C:\Users\ThinkPad\Documents\test sets\code\ph_smooth_5mins.csv')
#%%
def create_rmean_feats(w_list,series):
    series = series.copy()
    start = time.time()
    for w in w_list:
        newseries = series.rolling(window=w).mean().dropna()
        plt.plot(newseries)
    end = time.time()
    print(end-start) 
    return newseries
#%%
w_list = [100]
rwm_100_series = create_rmean_feats(w_list,series)

#%%
#找到时间序列的周期T；
#以T为分割点，对序列进行分割。假设序列的长度是n，分割后就会有n/T个单元；
#比较这n/T个单元的相似度，如果比较相似，则说明具有周期性，如果不是，则不具有周期性

def test_periodity(T,series):
    
    series_list = []
    dist_list = []
    leng = len(series)
    for i in range(0, len(series), T):
        series_list.append(series[i:i + T].values)
    #dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    for j in range(len(series_list)-2):
        x = series_list[j]
        y = series_list[j+1]
        dist = euclidean(x, y)
        dist_list.append(dist)
    return np.mean(dist_list)

#%%
print(test_periodity(10,rwm_100_series))
print(test_periodity(4,rwm_100_series))
print(test_periodity(1,rwm_100_series))
#%%
print(test_periodity(10,rwm_100_series))
print(test_periodity(4,rwm_100_series))
print(test_periodity(1,rwm_100_series))












































































