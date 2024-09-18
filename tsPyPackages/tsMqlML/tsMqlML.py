#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------
import numpy as np
import pandas as pd
#+-------------------------------------------------------------------
# import ai packages scikit learns
#+-------------------------------------------------------------------
#sklearn library. it includes utilities for data preprocessing, model evaluation, and model
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from scikit-learn, which is often used for cross-validat
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
#+--------------------------------------------------
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self, lp_df ,lv_X = [], lv_y = [] ,lv_X_train = [], lv_X_train_scaled =[],lv_X_test =[], lv_X_test_scaled = [],lv_y_train =[],lv_y_test = []):
        self._lp_df = lp_df
        
    @property
    def lp_df(self):
        return self.lp_df

    @lp_df.setter
    def lp_dfx(self, value):
        self._lp_df = value
                
    @property
    def lv_X(self):
        return self.lv_X
    @lv_X.setter
    def lv_X(self, value):
        value = self._lv_X = []
        self._lv_X = value
        
    @property
    def lv_y(self):
        return self.lv_y
    @lv_y.setter
    def lv_y(self, value):
        value = self._lv_y = []
        self._lv_y = value
        
    @property
    def lv_X_train(self):
        return self.lv_X_train
    @lv_X_train.setter
    def lv_X_train(self, value):
        self._lv_X_train = value   
        
    @property
    def lv_X_test(self):
        return self.lv_X_test
    @lv_X_test.setter
    def lv_X_test(self, value):
        self._lv_X_test = value      
        
    @property
    def lv_X_train_scaled(self):
        return self.lv_X_train_scaled
    @lv_X_train_scaled.setter
    
    def lv_X_train_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.fit_transform(self.lv_X_train)
        self._lv_X_train_scaled = value   
        
    @property
    def lv_X_test_scaled(self):
        return self.lv_X_test_scaled
    @lv_X_test_scaled.setter
    def lv_X_test_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.transform(self.lv_X_test)
        self._lv_X_test_scaled = value          
        
    @property
    def lv_y_train(self):
        return self.lv_y_train
    @lv_y_train.setter
    def lv_y_train(self, value):
        self._lv_y_train = value       

    @property
    def lv_y_test(self):
        return self.lv_y_test
    @lv_y_test.setter
    def lv_y_test(self, value):
        self._lv_y_test = value  
        
        
#--------------------------------------------------------------------
# create method  "dl_split_data_sets".
# class:cmqlmlsetup      
# usage: mql data
# \pdlsplit data
#--------------------------------------------------------------------
    def dl_split_data_sets(lp_df, lv_X = [],lv_y = [],lp_test_size=0.2,lp_shuffle = False, lp_prog = 1): 
        lv_X = []
        lv_y = []
        lv_X = lp_df[['close']]   
        lv_y = lp_df['target']
    
        lv_X_train,lv_y_train,lv_X_test,lv_y_test =  train_test_split(lv_X, lv_y, test_size=lp_test_size, shuffle=lp_shuffle)        
        if lp_prog == 1:
            return lv_X_train
        if lp_prog == 2:
            return lv_y_train     
        if lp_prog == 3:          
            return lv_X_test
        if lp_prog == 4:
            return lv_y_test            
#--------------------------------------------------------------------       
# create method  "dl_train_model_scaled".
# class:cmqlmlsetup      
# usage: mql data
# \param  var  
#-------------------------------------------------------------------- 
    def dl_train_model_scaled(lp_X_train):
        # meta names
        scaler = StandardScaler()
        lp_X_train_scaled = scaler.fit_transform(lp_X_train)
        return lp_X_train_scaled
    
#-------------------------------------------------------------------- 
# create method  "dl_test_model_scaled".
# class:cmqlmlsetup      
# usage: mql data
# \param  var  
#----------test----------------------------------------------------- 
    def dl_test_model_scaled(lp_X_test = []):
        # meta names
        scaler = StandardScaler()
        lp_X_test_scaled = scaler.fit_transform(lp_X_test)
        return lp_X_test_scaled   
        

#--------------------------------------------------------------------
# create method  "dl_build_neuro_network".
# class:cmqlmlsetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_build_neuro_network(lp_k_reg= 0.01,lv_X_train = 0):
        # Build a neural network model
        model=None
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(lv_X_train.shape[1],), kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(1, activation='linear'))
        return model

#--------------------------------------------------------------------  
# create method  "dl_compile_neuro_network".
# class:cmqlmlsetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_compile_neuro_network(lp_model,lp_optimizer = 'adam',lp_loss = 'mean_squared_error'):
        # Compile thelp_optimizer, lp_loss
        lv_model=lp_model.compile(optimizer=lp_optimizer, loss=lp_loss)
        return lv_model
    
# create method  "dl_train_model".
# class:cmqlmlsetup      
# usage: mql data
# \param  var  
#-------------------------------------------------------------------- 
    def dl_train_model(lp_df,lp_test_size=0.2,lp_shuffle=False,lv_X = 0,lv_y = 0):
        X_train, X_test, y_train, y_test = np.train_test_split(lv_X, lv_y, lp_test_size, lp_shuffle) 
        return X_train, X_test, y_train, y_test

#--------------------------------------------------------------------   
# create method  "dl_compile_neuro_network".
# class:cmqlmlsetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_train_neuro_network(lp_model,lp_X_train_scaled,lp_y_train,lp_epochs,lp_batch_size=256, lp_validation_split=0.2, lp_verbose=1):
        # Train the model
        lp_model.fit(lp_X_train_scaled, lp_y_train, lp_epochs, lp_batch_size, lp_validation_split, lp_verbose)
        return lp_model

#--------------------------------------------------------------------
# create method  "dl_predict_network"
# class:cmqlmlsetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_predict_network(lp_df, lp_model,lp_seconds = 60):         
        # Use the model to predict the next 4 instances
        lv_X = lp_df[['close']]
        lv_y = lp_df['target']     
            
        lv_X_predict=[]
        lv_X_predict_scaled=[]

        lv_predictions = pd.DataFrame()
        lv_predictions=[]
        scaler = StandardScaler()
        # Empty DataFrame
        lv_X_predict = lp_df.tail(lp_seconds)[['close']]
        lv_X_predict_scaled = scaler.transform(lv_X_predict)
        lv_predictions = lp_model.predict(lv_X_predict_scaled)
        return lv_predictions

#--------------------------------------------------------------------
# create method  "dl_model_performance"
# class:cmqlmlsetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
#Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
# The lower the MSE, the better the model.

#Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values. 
# Like MSE, lower values indicate better model performance.

#R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the 
# dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates 
# a perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the
#target variable. Negative values indicate poor model performance.
    def dl_model_performance(lp_y_test,lp_model,lp_X_test_scaled):
        # Calculate and print mean squared error
        mse = mean_squared_error(lp_y_test, lp_model.predict(lp_X_test_scaled))
        print(f"\nMean Squared Error: {mse}")
        # Calculate and print mean absolute error
        mae = mean_absolute_error(lp_y_test, lp_model.predict(lp_X_test_scaled))
        print(f"\nMean Absolute Error: {mae}")
        # Calculate and print R2 Score
        r2 = r2_score(lp_y_test, lp_model.predict(lp_X_test_scaled))
        print(f"\nR2 Score: {r2}")
        return r2