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

# test mode to pass through litnus test data
mp_test=True
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
# Address TensorFlow warning
tf.compat.v1.reset_default_graph()
# ======================================================
# import local packages
# ======================================================
# Check if GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
mp_rowcount = 100
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
mp_epochs=1
mp_batch_size=16
mp_num_trials=1
mp_num_models=1
# End Params


mt=CMdtuner
# Run the tuner
# Start Params
mp_objective='val_accuracy'
mp_max_trials=10  # Number of hyperparameter sets to try
mp_executions_per_trial=1  # Number of models to build and evaluate for each trial
mp_modeldatapath = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\PythonLib\tsModelData"
mp_directory=mp_modeldatapath + '\\' + 'tshybrid_ensemble_tuning'
mp_project_name=mp_modeldatapath + '\\' + 'tshybrid_ensemble_model'
print("mp_directory",mp_directory)     
print("mp_project_name",mp_project_name)
# End Params


 # Truncate 'x' to match 'y'
mv_X_train_scaled = mv_X_train_scaled[:len(mv_y_train)] 
mv_X_test_scaled = mv_X_test_scaled[:len(mv_y_test)] 

mp_train_input_shape =shape=(mv_X_train_scaled.shape)
mp_test_input_shape = shape=(mv_X_test_scaled.shape)
print(f"mp_train_input_shape: {mp_train_input_shape}")
print(f"mp_test_input_shape: {mp_test_input_shape}")

############################################
# Start Test Load Data
############################################

if mp_test == True:
    mv_X_train_scaled=np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
    mv_X_test_scaled=np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature

    mv_y_train = np.random.randint(2, size=(1000,))  # Binary target
    mv_y_test = np.random.randint(2, size=(1000,))  # Binary target
    
    mp_train_input_shape = shape=(100, 1)
    mp_test_input_shape =  shape=(100, 1)
############################################
#End Test Load Data
############################################

# Print input shapes
print(f"mv_X_train_scaled shape: {mv_X_train_scaled.shape}")
print(f"mv_y_train shape: {mv_y_train.shape}")
print(f"mp_train_input_shape: {mp_train_input_shape}")
print(f"mp_test_input_shape: {mp_test_input_shape}")

# Ensure mp_epochs and mp_batch_size are integers
print(f"mp_epochs: {mp_epochs}, type: {type(mp_epochs)}")
print(f"mp_batch_size: {mp_batch_size}, type: {type(mp_batch_size)}")

# Run tuner
best_model = mt.run_tuner(mp_train_input_shape, mv_X_train_scaled, mv_y_train, mp_objective, mp_max_trials, mp_executions_per_trial, mp_directory, mp_project_name, mp_validation_split, mp_epochs, mp_batch_size)

# Print the summary of the best model
best_model.summary()

# Check the expected input shape
expected_input_shape = [(mp_train_input_shape), (mp_train_input_shape), (mp_train_input_shape), (mp_train_input_shape)]
print("Expected Shape full load:", expected_input_shape)


if mp_test == True:
        # Check the expected input shape
        expected_input_shape = [(None, 100, 1), (None, 100, 1), (None, 100, 1), (None, 100, 1)]
        print("Expected Shape Test mode:", expected_input_shape)

# Verify the shape of your training data
print("Original mv_X_train_scaled shape:", mv_X_train_scaled.shape)

# Reshape the training data to match the expected input shape
mv_X_train_scaled_list = [np.reshape(mv_X_train_scaled, (-1, shape[1], shape[2])) for shape in expected_input_shape]
print("Reshaped mv_X_train_scaled shapes:", [x.shape for x in mv_X_train_scaled_list])

# Verify the shape of your test data
print("Original mv_X_test_scaled shape:", mv_X_test_scaled.shape)

# Reshape the test data to match the expected input shape
mv_X_test_scaled_list = [np.reshape(mv_X_test_scaled, (-1, shape[1], shape[2])) for shape in expected_input_shape]
print("Reshaped mv_X_test_scaled shapes:", [x.shape for x in mv_X_test_scaled_list])

# Create a list of exactly 4 identical tensors
mv_X_train_list = [mv_X_train_scaled] * 4

# Correct the call to best_model.fit
mv_model = best_model.fit(mv_X_train_list, mv_y_train, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)

# Create a list of exactly 4 identical tensors
mv_X_test_list = [mv_X_test_scaled] * 4

# Predict the model
# Create a list of exactly 4 identical tensors
mv_X_test_list = [mv_X_test_scaled] * 4

# Predict the model
predictions = best_model.predict(mv_X_test_list)

print("Predictions:", predictions)

# Model performance
accuracy, precision, recall, f1 = m1.model_performance(best_model, mv_X_test_list, mv_y_test)
