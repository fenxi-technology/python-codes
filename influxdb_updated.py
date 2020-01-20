#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
Created on Mon Jan 13 16:22:58 2020

@author: ThinkPad
"""
import pandas as pd
from influxdb import InfluxDBClient
import time
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

class operate_influxdb:

    def __init__(self, database, ip='localhost', port=8086, username='admin', password='admin'):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.database = database

    def get_connect(self):
        conn = InfluxDBClient(self.ip, self.port, self.username, self.password, self.database)
        return conn
    

    def query(self, sql):
        conn = self.get_connect()
        result = conn.query(sql)
        return result
    
    def query_df(self,sql,tablename):
        conn = self.get_connect()
        result = conn.query(sql)
        df = pd.DataFrame(list(result.get_points(measurement=tablename)))
        return df

    def query_all_df(self, tablename):
        conn = self.get_connect()
        sql = 'select * from ' + '"%s"' %(tablename) + ';'
        result = conn.query(sql)
        df = pd.DataFrame(list(result.get_points(measurement=tablename)))
        return df

    
    def insert_points(self, point_dict_list, tablename):
        conn = self.get_connect()
        json_body = []
        for i in range(len(point_dict_list)):
            a = {
                "measurement": tablename,
                "tags": {"id": i},
                "fields": point_dict_list[i]
            }
            json_body.append(a)
        conn.write_points(json_body)

    
    def insert_df(self, df, tablename):
        conn = self.get_connect()
        length = len(df)
        batch = 10000
        n = int(length / batch + 1)
        rem = length % batch

        for j in range(n):
            if j != n - 1:
                json_body = []
                for i in range(batch):
                    index = j * batch + i
                    a = {
                        "measurement": tablename,
                        "tags": {"id": index},
                        "fields": df.iloc[index, :].to_dict(),
                        "time": df.Datum[index]
                    }
                    json_body.append(a)
                conn.write_points(json_body)
            else:
                json_body = []
                for i in range(rem):
                    index = j * batch + i
                    a = {
                        "measurement": tablename,
                        "tags": {"id": index},
                        "fields": df.iloc[index, :].to_dict(),
                        "time": df.Datum[index]
                    }
                    json_body.append(a)
                conn.write_points(json_body)

                
    def show_all_tables(self):
        sql = 'show measurements;'
        return self.query(sql)

    def drop_table(self, tablename):
        sql = 'drop measurement ' + tablename + ';'
        self.query(sql)

    def delete_table_data(self, tablename):
        sql = 'delete from ' + tablename + ';'
        self.query(sql)




#%%
#test
if __name__ == "__main__":
    om = operate_influxdb('myDB')
    print(om.show_all_tables())
    #lsj = om.query_df_python_format('select * from "0拉丝机"','0拉丝机')
    #lsj = om.query_all_df_python_format('0拉丝机')
    lsj_null = om.query_all_df('0拉丝机')
    #jsj = om.query_all_df_python_format('10挤塑机')
    #cjy = om.query_all_df_python_format('11测径仪')
    #ccy = om.query_all_df_python_format('12测长仪')
    #jxj = om.query_all_df_python_format('1绞线机')
    #jlj1 = om.query_all_df_python_format('2交联机1')
    #jlj2 = om.query_all_df_python_format('3交联机2')
    #pxj1 = om.query_all_df_python_format('4偏心1')
    #pxj2 = om.query_all_df_python_format('5偏心2')
    #tp1 = om.query_all_df_python_format('6铜屏1')
    #tp2 = om.query_all_df_python_format('7铜屏2')
    #kz = om.query_all_df_python_format('8铠装')
    #cl = om.query_all_df_python_format('9成缆')
    #cjy = om.query_df_python_format('select * from "11测径仪"','11测径仪')
#%%
def influxdb_df_to_python_df_update(tablename):
    #merge on actual time
    #tablename = '0拉丝机'
    om = operate_influxdb('myDB')
    codelist = om.query_df('select * from "%s" limit 100'%tablename,tablename).CodeName.unique()
    sql = 'select * from "%s" where CodeName=%s;' % (tablename, "'%s'"%codelist[0])
    lsj_0 = om.query_df(sql,tablename)
    fields = codelist[1:]

    lsj = lsj_0
    lsj['Time'] = lsj['Date'].str.cat(lsj['Time'],sep=' ').apply(lambda x: pd.Timestamp(x))
    for i in fields:
        sql = 'select Date,Time,CodeName,Value from "%s" where CodeName=%s;' % (tablename, "'%s'" %i) 
        lsj_partial = om.query_df(sql,tablename)
        lsj_partial['Time'] = lsj_partial['Date'].str.cat(lsj_partial['Time'],sep=' ').apply(lambda x: pd.Timestamp(x)) 
        lsj = lsj.merge(lsj_partial,on='Time',how='outer')
    
    lsj.drop(['time_x','time_y','Date_x','Date_y'],axis=1,inplace=True)  
    code1 = lsj.CodeName_x
    valuex = lsj.Value_x
    code2 = lsj.CodeName_y
    valuey = lsj.Value_y
    valuex.columns = code1.iloc[0].values
    valuey.columns = code2.iloc[0].values
    
    new = lsj.drop(['CodeName_x','CodeName_y','Value_x','Value_y'],axis=1)
    new = pd.concat([new,valuex,valuey],axis=1)
    
    if 'CodeName' in new.columns:
        new.rename(columns={'Value':'%s'%new.CodeName[0]},inplace=True)
        new.drop('CodeName',axis=1,inplace=True)
    
    if 'time' in new.columns:
        new.drop('time',axis=1,inplace=True)
        
    if 'Date' in new.columns:
        new.drop('Date',axis=1,inplace=True)
        
    #change sequence
    new_time = new.Time
    new = new.drop('Time',axis=1)
    new.insert(0,'Time',new_time)
    
    return new

#%%
#insert my db
file = r'C:\Users\ThinkPad\Documents\test sets\DrHartmann project\Dr_Hartmann_Project\52692_MWS_output.csv'
df = pd.read_csv(file, usecols=['KK1_ABS1_PH1'])   

if __name__ == "__main__":
    om = operate_influxdb('testdataMWS')
    for i in range(len(df)):
        point = [{'KK1_ABS1_PH1': df.iloc[i,0]}]
        om.insert_points(point, 'PH')
        time.sleep(1)
        print(i)
    #om.query('drop measurement PH;')
    #data = om.query_df('select * from Raw_MWS_with_time;', 'Raw_MWS_with_time')
    #print(data.head())
    













