# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from math import sqrt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
#%%
#import data
dd_MWS = pd.read_csv(r'C:\Users\ThinkPad\Documents\test sets\DrHartmann project\Dr_Hartmann_Project\52692_MWS_output.csv').iloc[:,:27]
print(dd_MWS.shape)


#%%
class neural_network():
    
    
    
    def __init__(self, data, target_col='KK1_ABS1_PH1', **kwargs):
        
        self.data = data
        self.target_col = target_col
        self.k = kwargs.get('k',50)
        self.pca_ratio = kwargs.get('pca_ratio',0.99)
        self.method = kwargs.get('method','rf')
        self.train_ratio = kwargs.get('train_ratio',0.6)
        self.look_back = kwargs.get('look_back',15)
        self.lead_time = kwargs.get('lead_time',15)
        self.output_dim = kwargs.get('output_dim', 18)
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')
        self.activation_dense = kwargs.get('activation_dense', 'relu')
        self.activation_last = kwargs.get('activation_last', 'sigmoid')    
        self.dense_layer = kwargs.get('dense_layer', 1)     
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.5)
        self.nb_epoch = kwargs.get('nb_epoch', 12)
        self.batch_size = kwargs.get('batch_size', 72)
        self.loss = kwargs.get('loss', 'mse')
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.x_train = []  # 初始化训练集x部分-训练特征
        self.y_train = []  # 初始化训练集y部分-监督信号
        self.x_test = []  # 初始化测试集x部分-测试特征
        self.y_test = []
    
    
    
    def feature_selection_correlation(self,df):
        
        k = self.k
        
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        #select features with high correlation
        ch2 = SelectKBest(chi2, k)
        X_1 = ch2.fit_transform(X, y.astype('int'))
        list_ = ch2.get_support(indices=True).tolist()
        df1 = pd.DataFrame(X_1,columns=[df.columns[i] for i in list_])
        df1 = pd.concat([df1, y], axis=1)
        df1 = df1.rename(columns = {0:self.target_col})
    
        print('Columns after selections are',df1.columns)
        
        return df1
    
    
    
    def rf_importance(self,df):
    
        k = self.k
        
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        
        rf = RandomForestRegressor()
        rf.fit(X, y)
        
        ranking = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), df.columns.to_list()), reverse=True)[:k]
        print(ranking)
        
        co_list = []
        for i in range(k):
            co_list.append(ranking[i][1])
        
        print('Columns after selections are',co_list)

        return df[co_list]
 
    
    
    def rfe_function(self,df):
    
        k = self.k
        
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        
        lr = Ridge(alpha=100000, fit_intercept=True, normalize=True, copy_X=True, max_iter=1500, tol=1e-4, solver='auto')
        rfe = RFE(estimator=lr, n_features_to_select=k)
        rfe.fit_transform(X, y)
        ranking = sorted(zip(rfe.ranking_,X.columns.to_list()), reverse=True)[:k]
    
        co_list = []
        for i in range(k):
            co_list.append(ranking[i][1])
            
        print('Columns after selections are',co_list)

        return df[co_list]
        
    
    
    def preprocess(self):
        
        data = self.data.dropna()
        
        print('number of columns before feat selection is',self.data.shape[1])
        k = self.k
        pca_ratio = self.pca_ratio
        method = self.method
        
        #define X&y
        y = data[self.target_col]
        X = data[data.columns.difference([self.target_col])]
        
        #normalize data
        self.scaler = MinMaxScaler(feature_range=(0, 1))  
        self.data = self.scaler.fit_transform(np.c_[X,y])
        
        y_scaled = pd.DataFrame(self.data[:,-1])
        X_scaled = self.data[:,:-1]
        
        #convert array to df to fit in feature_selection_correlation and rf_importance function
        df2 = pd.DataFrame(X_scaled,columns=[data.columns[i] for i in range(data.shape[1]-1)])
        df2 = pd.concat([df2, y_scaled], axis=1)
        df2 = df2.rename(columns = {0:self.target_col})

        global df
        
        if method == 'corr':
            df = self.feature_selection_correlation(df2)
        if method == 'rf':
            df = self.rf_importance(df2)
        if method == 'rfe':
            df = self.rfe_function(df2)
        if method == 'pca':
            pca = PCA(n_components=pca_ratio)
            X_pca = pca.fit_transform(X_scaled)
            print(pca.explained_variance_ratio_)
            print(pca.explained_variance_)
            df = pd.DataFrame(np.c_[X_pca,y_scaled])    
        
        print('number of columns after feat selection is',df.shape[1])
        return df.values
    
        
    
    def split_dataset(self):
        
        data = self.preprocess()
        
        def create_dataset(data, look_back, lead_time):
            dataX, dataY = [], []
            for i in range(len(data) - look_back - lead_time):
                a = data[i:(i + look_back), :]
                dataX.append(a)
                dataY.append(data[i + look_back + lead_time, -1])
            return np.array(dataX), np.array(dataY)


        train_size = int(len(data) * self.train_ratio)
        self.train_data = data[:train_size, :] #6000,形成了5970个15
        self.test_data = data[train_size - self.look_back - 1:len(data), :] #5984到10000
        #？
        # 具体分割后数据集
        #x_all, self.y_true = create_dataset(self.data, self.look_back)
        x_train, self.y_train = create_dataset(self.train_data, self.look_back, self.lead_time)
        x_test, self.y_test = create_dataset(self.test_data, self.look_back, self.lead_time)

        # Reshape input to be [samples, time_step, features]
        #self.x_train = np.reshape(x_train, (x_train.shape[0], self.look_back, x_train.shape[2]))
        #self.x_test = np.reshape(x_test, (x_test.shape[0], self.look_back, x_test.shape[2]))
        self.x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2],self.look_back))
        self.x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2],self.look_back))
        
        print(self.x_train.shape[0],self.x_train.shape[1],self.x_train.shape[2])
        print(self.x_test.shape[0],self.x_test.shape[1],self.x_test.shape[2])
        print(self.y_train.shape)
        print(self.y_test.shape)


    
    def inverse_transform(self,trainPredict,testPredict):
        
        train_size = int(len(self.data) * self.train_ratio)
        x_train0 = self.data[:train_size - self.look_back - self.lead_time, :-1]
        x_test0 = self.data[train_size - 1 -self.look_back : len(self.data)-self.lead_time-self.look_back,:-1]
        print(x_test0.shape)
        y_train = self.y_train.reshape(len(self.y_train),1)
        y_test = self.y_test.reshape(len(self.y_test),1)
        
        # 将预测值反标准化到正常值
        yhat_train = trainPredict.reshape(len(trainPredict),1)
        # invert scaling for forecast
        inv_yhat_train = np.concatenate((x_train0,yhat_train), axis=1)
        inv_yhat_train = self.scaler.inverse_transform(inv_yhat_train)
        inv_yhat_train = inv_yhat_train[:,-1]
        # invert scaling for actual
        inv_y_train = np.concatenate((x_train0, y_train), axis=1)
        inv_y_train = self.scaler.inverse_transform(inv_y_train)
        inv_y_train = inv_y_train[:,-1]

        yhat_test = testPredict.reshape(len(testPredict),1)
        # invert scaling for forecast
        inv_yhat_test = np.concatenate((x_test0,yhat_test), axis=1)
        inv_yhat_test = self.scaler.inverse_transform(inv_yhat_test)
        inv_yhat_test = inv_yhat_test[:,-1]
        # invert scaling for actual
        inv_y_test = np.concatenate((x_test0, y_test), axis=1)
        inv_y_test = self.scaler.inverse_transform(inv_y_test)
        inv_y_test = inv_y_test[:,-1]
        
        return inv_yhat_train,inv_y_train,inv_yhat_test,inv_y_test

 
    
    def NN_model(self):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: epect value of test data
        :return: model after training
        """
        print("Training model is LSTM network!")
        
        
        input_dim = self.x_train[2]
        input_shape = (None,self.look_back)
        #input_shape = (self.look_back,self.x_train[2])
        #output_dim = trainY.shape[1] # one-hot label
        output_dim = 1
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfitting
        model.add(LSTM(units=self.output_dim,
                       input_shape=input_shape,
                       activation=self.activation_lstm,
                       recurrent_dropout=self.drop_out,
                       return_sequences=True))
        for i in range(self.lstm_layer-2):
            model.add(LSTM(units=self.output_dim,
                       input_shape=input_shape,
                       activation=self.activation_lstm,
                       recurrent_dropout=self.drop_out,
                       return_sequences=True))
        # argument return_sequences should be false in last lstm layer to avoid input dimension incompatibility with dense layer
        model.add(LSTM(units=self.output_dim,
                       input_shape=input_shape,
                       activation=self.activation_lstm,
                       recurrent_dropout=self.drop_out))
        for i in range(self.dense_layer-1):
            model.add(Dense(units=self.output_dim,
                        activation=self.activation_last))
        model.add(Dense(units=output_dim,
                        activation=self.activation_last))
        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer)
        # train the model with fixed number of epoches
        history = model.fit(x=self.x_train,y=self.y_train, nb_epoch=self.nb_epoch, batch_size=self.batch_size)
        score = model.evaluate(self.x_train, self.y_train, self.batch_size)
        print("Model evaluation: {}".format(score))
        
        # LSTM prediction/LSTM进行预测
        trainPredict = model.predict(self.x_train)  # Predict by training data set
        testPredict = model.predict(self.x_test)  # Predict by test data set

        inv_yhat_train, inv_y_train, inv_yhat_test, inv_y_test = self.inverse_transform(trainPredict,testPredict)
        # calculate RMSE
        train_rmse = sqrt(mean_squared_error(inv_y_train, inv_yhat_train))
        print('TRAIN RMSE: %.3f' % train_rmse)
        plt.plot(inv_y_train)
        plt.plot(inv_yhat_train)
        plt.show()
        
        test_rmse = sqrt(mean_squared_error(inv_y_test, inv_yhat_test))
        print('TEST RMSE: %.3f' % test_rmse)
        plt.plot(inv_y_test)
        plt.plot(inv_yhat_test)
        plt.show()
    
    
    
    
#%% 
def main():
    obj_NN = neural_network(data=ds.iloc[:10000,1:])
    obj_NN.split_dataset()
    model = obj_NN.NN_model()
    
if __name__ == "__main__":
   main()
#%%
