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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

from tabulate import tabulate

from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup
from tsMqlMLTune import CMdtuner

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
import keras_tuner as kt
# ======================================================
# import local packages
# ======================================================



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
mp_rows = 1000
mp_rowcount = 1000
mp_command = mt5.COPY_TICKS_ALL
mp_dfName = "df_rates"
mv_manual = True
mp_path = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\Mql5Data"

lpfileid="tickdata1"
mp_filename = mp_symbol_primary + "_" + lpfileid + ".csv"
# End Params

d1 = CMqldatasetup
# create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
mv_utc_from = d1.set_mql_timezone(mp_year, mp_month, mp_day, mp_timezone)

print("Timezone Set to : ", mv_utc_from)
print("mp_path Set to : ", mp_path)
print("mp_filename Set to : ", mp_filename)
mv_ticks1 = pd.DataFrame(d1.run_load_from_mql(mv_manual,mp_dfName,mv_utc_from,mp_symbol_primary,mp_rows,mp_rowcount,mp_command,mp_path,mp_filename))
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
#no time                  bid      ask     last      volume      time_msc        flags       volume_real
#0 2020-01-10 00:00:00  1.11051  1.11069   0.0       0           1578614400987    134          0.0
#1 2020-01-10 00:00:02  1.11049  1.11067   0.0       0           1578614402025    134          0.0

mp_seconds = 60
mp_unit = 's'
# End Params
print("cols tick2:",mv_ticks2.columns)

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
mt = CMdtuner
# start Params


mp_k_reg = 0.001
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
mp_seq=False
mp_input_dim=1
mp_output_dim=1
mp_filt=64
mp_ksize=3
mp_k_reg = 0.01
mp_pool=2
mp_optimizer = 'adam'
mp_act1='relu'
mp_act2='linear'
mp_act2='linear'
mp_act3='sigmoid'
mp_metric = 'accuracy'
mp_loss1 = 'mean_squared_error'
mp_loss2 = 'binary_crossentropy'
mp_objective='val_accuracy'

mp_cells1=128
mp_cells2=256
mp_cells3=128
mp_cells4=64
mp_cells5=1

mp_min=32
mp_max=128
mp_step=2
mp_hmin=8
mp_hmax=128
mp_hstep=2
mp_imin=64
mp_imax=256
mp_istep=64
mp_jmin=0.5
mp_jmax=0.2
mp_jstep=0.1

mp_validation_split=0.2
mp_epochs=10
mp_batch_size=32
mp_num_trials=1
mp_num_models=1
# End Params


mt=CMdtuner
# Run the tuner
# Start Params
mv_objective='val_accuracy'
mv_max_trials=10  # Number of hyperparameter sets to try
mv_executions_per_trial=1  # Number of models to build and evaluate for each trial
mv_modeldatapath = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\Mql5Data\PythonLib\tsModelData"
mv_directory=mv_modeldatapath + '\\' + 'tshybrid_ensemble_tuning'
mv_project_name=mv_modeldatapath + '\\' + 'tshybrid_ensemble_model'
print("mv_directory",mv_directory)     
print("mv_project_name",mv_project_name)
# End Params

#pd.shape  # This returns the shape as a tuple, e.g., (rows, columns)
# If you want to access the number of rows or columns specifically:
#pd.shape[0]  # This will give you the number of rows
#pd.shape[1]  # This will give you the number of columns
# Ensure mv_X_train and mv_y_train are correctly shaped

 # Truncate 'x' to match 'y'
mv_X_train_scaled = mv_X_train_scaled[:len(mv_y_train)] 

print(f"mv_X_train_scaled shape: {mv_X_train_scaled.shape}")
print(f"mv_y_train shape: {mv_y_train.shape}")


# Ensure mp_epochs and mp_batch_size are integers
print(f"mp_epochs: {mp_epochs}, type: {type(mp_epochs)}")
print(f"mp_batch_size: {mp_batch_size}, type: {type(mp_batch_size)}")
objective='val_accuracy',


best_model = mt.run_tuner(mv_X_train_scaled.shape, mv_X_train_scaled, mv_y_train, mv_objective, mv_max_trials, mv_executions_per_trial, mv_directory, mv_project_name)

# Print the summary of the best model
best_model.summary()

# Optionally: train the best model on the full dataset

# Correct the call to best_model.fit
results = best_model.fit(
    mv_X_train_scaled,  # Assuming mv_X_train is already the correct input
    mv_y_train,  # Assuming mv_y_train is already the correct output
    epochs=mp_epochs,
    batch_size=mp_batch_size
)


print("Model Result:",results.history)


"""
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

In ge

# modelperformance
#m1.dl_model_performance(mv_model,mv_X_train_scaled, mv_X_test_scaled)
"""