# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    tony shepherd |
# |                                                  www.equinox.com |
# +------------------------------------------------------------------+
# property copyright "tony shepherd"
# property link      "www.equinox.com"
# property version   "1.01"
# +-------------------------------------------------------------------
# import standard python packages
# +-------------------------------------------------------------------
""" 
The model is initialized as a sequential model, meaning it's a linear stack of layers.
The Dense layers represent fully connected layers in the neural network. 
The parameters include the number of neurons (units), activation function, 
input shape (applicable only for the first layer), and kernel regularization 
(in this case, L2 regularization with a strength specified by k_reg ).
activation='relu' implies the Rectified Linear Unit (ReLU) activation function, commonly used in hidden layers.
The last layer has one neuron with a linear activation function, indicating a regression output.
If this were a classification task, a different activation function like 'sigmoid' or 'softmax' would be used.
This architecture is a feedforward neural network with multiple hidden layers for regression purposes. 
Adjust the parameters and layers based on your specific task and dataset characteristics.
"""

import sys
import datetime
from datetime import datetime
import pytz
import keyring as kr
# +-------------------------------------------------------------------
# import mql packages
# +-------------------------------------------------------------------
import MetaTrader5 as mt5
# numpy is widely used for numerical operations in python.
import numpy as np
import pandas as pd
import tabulate
import tsMqlConnect
import tsMqlData
import tsMqlML

from tabulate import tabulate
from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
import keras_tuner as kt
# ======================================================
# import local packages
# ======================================================

# User can use the alias if they want

mv_debug = 0

# ======================================================
# Start Login MQL Terminal
# ======================================================
# Add to hash Vault keyring externally via CLI

cred = kr.get_credential("xercesdemo", "")
c1 = CMqlinitdemo
# start Params
c1.path = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
c1.login = int(cred.username)
c1.password = str(cred.password)
c1.server = r"ICMarketsSC-Demo"
c1.timeout = 60000
c1.portable = True
# End Params

# =================
# Login Metatrader
# ======================================================
c1.run_mql_login(c1.path, c1.login, c1.password,
                 c1.server, c1.timeout, c1.portable)
# ======================================================
# End Login MQL Terminal
# ======================================================

# +-------------------------------------------------------------------
# Import Data from MQL
# +-------------------------------------------------------------------
# start Params
mp_symbol_primary = "EURUSD"
mp_symbol_secondary = "GBPUSD"
mp_year = 2024
mp_month = 1
mp_day = 1
mp_timezone = 'etc/UTC'
mp_rows = 100000
mp_command = mt5.COPY_TICKS_ALL
mp_dfName = "df_rates"
# End Params

d1 = CMqldatasetup
# create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
mv_utc_from = d1.set_mql_timezone(mp_year, mp_month, mp_day, mp_timezone)

print("Timezone Set to : ", mv_utc_from)
mv_ticks1 = pd.DataFrame(d1.run_load_from_mql(mv_debug, mp_dfName, mv_utc_from, mp_symbol_primary, mp_rows, mp_command))

# +-------------------------------------------------------------------
# Prepare Data
# +-------------------------------------------------------------------
mv_ticks2 = pd.DataFrame()  # Empty DataFrame
mv_ticks2 = mv_ticks2.drop(index=mv_ticks2.index)
mv_ticks2 = pd.DataFrame(mv_ticks1)

# +-------------------------------------------------------------------
# Shift Data
# +-------------------------------------------------------------------
# Tabulate formatting
# start Params
mp_seconds = 7200
mp_unit = 's'
# End Params

mv_ticks3 = d1.run_shift_data1(mv_ticks2, mp_seconds, mp_unit)

print(tabulate(mv_ticks1.head(10), showindex=False, headers=mv_ticks1.columns,tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks2.head(10), showindex=False, headers=mv_ticks2.columns,tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks3.head(10), showindex=False, headers=mv_ticks3.columns,tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))

print("mv_ticks3: ", len(mv_ticks3))
print("mv_ticks3:close ", len(mv_ticks3[['close']]))
print("mv_ticks3:target ", len(mv_ticks3[['target']]))


m1 = CMqlmlsetup

# Start Params
mp_test_size = 0.2
mp_shuffle = False

# End Params
mv_X_train = []
mv_y_train = []


mv_X_train = m1.dl_split_data_sets(mv_ticks3,mv_ticks3, mp_test_size, mp_shuffle, 1)
mv_y_train = m1.dl_split_data_sets(mv_ticks3,mv_ticks3, mp_test_size, mp_shuffle, 2)

mv_X_test = m1.dl_split_data_sets(mv_ticks3,mv_ticks3, mp_test_size, mp_shuffle, 3)
mv_y_test = m1.dl_split_data_sets(mv_ticks3,mv_ticks3, mp_test_size, mp_shuffle, 4)

mv_X_train_scaled = m1.dl_train_model_scaled(mv_X_train)
mv_X_test_scaled = m1.dl_test_model_scaled(mv_X_test)


# +-------------------------------------------------------------------
# Pre-tune a neural network model
# +-------------------------------------------------------------------
# start Params
mp_k_reg = 0.001
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
# End Params


mv_model1 = m1.dl_model_tune_build(mv_X_train_scaled,mv_X_test_scaled, mp_optimizer, mp_loss,mp_k_reg,)
mv_model2 = m1.dl_model_tune_run(mv_X_train_scaled,mv_X_test_scaled, mp_optimizer, mp_loss,mp_k_reg,)

print("Tune result:",mv_model2)

# +-------------------------------------------------------------------
# Build a neural network model
# +-------------------------------------------------------------------
# start Params
mp_k_reg = 0.01
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
# End Params
mv_model = m1.dl_build_neuro_network(mp_k_reg, mv_X_train, mp_optimizer, mp_loss)
# +--------------------------------------------------------------------
# Train the model
# +--------------------------------------------------------------------
# start Params
mp_epoch = 5
mp_batch_size = 256
mp_validation_split = 0.2
mp_verbose = 0
# End Params

# Correct the size difference until reasoon known
mv_y_train_index=len(mv_y_train)
mv_y_train=mv_y_train.drop(mv_y_train_index)
print("mv_X_train",len(mv_X_test))
print("mv_y_train",len(mv_y_train))
print("mv_X_train-scaled",len(mv_X_test_scaled))

mv_model = m1.dl_train_model(mv_model, mv_X_train_scaled, mv_y_train,mp_epoch, mp_batch_size, mp_validation_split, mp_verbose)

# +--------------------------------------------------------------------
# Predict the model
# +--------------------------------------------------------------------
# start Params
mp_seconds=6000
# End Params
#print("MvTick3Tail:",mv_ticks3.tail(mp_seconds)['close'].values)
mv_predictions=m1.dl_predict_values(mv_ticks3,mv_model,mp_seconds)

# Print actual and predicted values for the next  n  instances
print("Actual Value for the Last Instances:")
print(mv_ticks2.tail(1)['close'].values)

print("\nPredicted Value for the Next Instances:")
print(mv_predictions[:, 0])
df_predictions=pd.DataFrame(mv_predictions)

"""
Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
The lower the MSE, the better the model.
Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values.
Like MSE, lower values indicate better model performance.

R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the 
dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a perfect
fit, while a score of 0 suggests that the model is no better than predicting the mean of the target variable. 
Negative values indicate poor model performance.  

for example use 900 days of data and 1 epoch for EURUSD for a 2 hours time period

R2 (R-squared):

1: R2 is a statistical measure that represents the proportion of the variance in the dependent variable that is
predictable from the independent variable(s).It is a scale from 0 to 1, where 0 indicates that the model does not
explain the variability of the dependent variable, and 1 indicates perfect explanation.

2:MSE (Mean Squared Error):MSE is a metric that calculates the average squared difference between the actual
and predicted values.It penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.
Mathematically, it is the sum of the squared differences divided by the number of observations.
MAE (Mean Absolute Error):

3:MAE is a metric that calculates the average absolute difference between the actual and predicted values.
Unlike MSE, it does not square the differences, making it less sensitive to outliers.
It is the sum of the absolute differences divided by the number of observations.

In general, a higher R2 value and lower MSE or MAE values indicate a better-performing model.

"""

# modelperformance
#m1.dl_model_performance(mv_model,mv_X_train_scaled, mv_X_test_scaled)

#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#y_true = mv_ticks3[['target']].values
#y_pred = mv_predictions[:, 0]

#mse = mean_squared_error(y_true, y_pred)
#mae = mean_absolute_error(y_true, y_pred)
#r2 = r2_score(y_true, y_pred)

#print(f"MSE: {mse}, MAE: {mae}, R²: {r2}"