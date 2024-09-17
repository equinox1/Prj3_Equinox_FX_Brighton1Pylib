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
# Class CMqlml
#+--------------------------------------------------
class CMqlml:
    def __init__(self, x=None, y=None):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self.x or self.defaultX()

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self.y or self.defaultY()

    @y.setter
    def y(self, value):
        self._y = value
        
#--------------------------------------------------------------------       
# create method  "dl_split_features_and_target".
# class: cmqldatasetup      
# usage: mql data
# \param  var  
#-------------------------------------------------------------------- 
    def dl_split_features_and_target(lp_df,lp_test_size=0.2,lp_shuffle=False):
        # Split the data into features (X) and target variable (y)
        X=[]
        y=[]
        X = lp_df[['close']]
        y = lp_df['target'] 
    
        # Split the data into training and testing sets
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        X_train, X_test, y_train, y_test = np.train_test_split(X, y, lp_test_size, lp_shuffle)
    
        # Standardize the features
        X_train_scaled=[]
        X_test_scaled=[]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train

#--------------------------------------------------------------------
# create method  "dl_build_neuro_network".
# class: cmqldatasetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_build_neuro_network(lp_k_reg,lp_X_train):
        # Build a neural network model
        model=None
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(lp_X_train.shape[1],), kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(lp_k_reg)))
        model.add(Dense(1, activation='linear'))
        return model

#--------------------------------------------------------------------  
# create method  "dl_compile_neuro_network".
# class: cmqldatasetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_compile_neuro_network(lp_model,lp_optimizer = 'adam',lp_loss = 'mean_squared_error'):
        # Compile thelp_optimizer, lp_loss
        lp_model.compile(lp_optimizer, lp_loss)
        return lp_model

#--------------------------------------------------------------------   
# create method  "dl_compile_neuro_network".
# class: cmqldatasetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_train_neuro_network(lp_model,lp_X_train_scaled,lp_y_train,lp_epochs,lp_batch_size=256, lp_validation_split=0.2, lp_verbose=1):
        # Train the model
        lp_model.fit(lp_X_train_scaled, lp_y_train, lp_epochs, lp_batch_size, lp_validation_split, lp_verbose)

#--------------------------------------------------------------------
# create method  "dl_predict_network"
# class: cmqldatasetup      
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_predict_network(lp_df, lp_model,lp_seconds = 60):         
        # Use the model to predict the next 4 instances
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
# class: cmqldatasetup      
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