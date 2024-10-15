# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    tony shepherd |
# |                                                  www.equinox.com |
# +------------------------------------------------------------------+
# property copyright "tony shepherd"
# property link      "www.equinox.com"
# property version   "1.01"
# test mode to pass through litnus test data
mp_test=False

# +-------------------------------------------------------------------
# import standard python packages
# +-------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keyring as kr
from tabulate import tabulate


# import mql packages
# +-------------------------------------------------------------------
import MetaTrader5 as mt5
# numpy is widely used for numerical operations in python.
import numpy as np
import pandas as pd

from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup
from tsMqlMLTune import CMdtuner

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf

# Ensure compatibility with TensorFlow v1 functions
tf.compat.v1.reset_default_graph()

# ======================================================
# import local packages
# ======================================================
# Check if GPU is available and set memory growth
gpus = tf.config.list_physical_devices('GPU')
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

# start Params
MPPATH = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
MPLOGIN = int(cred.username) # Ensure username is treated as a string
MPPASS = str(cred.password)
MPSERVER = r"ICMarketsSC-Demo"
MPTIMEOUT = 60000
MPPORTABLE = True


# Create an instance of the class

c1 = CMqlinitdemo(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
# End Params

# =================
# Login Metatrader
# ======================================================
c1.run_mql_login(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
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
mp_rowcount = 100 #560 

mp_command = mt5.COPY_TICKS_ALL
mp_dfName = "df_rates"
mv_manual = True
mp_path = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\Mql5Data"

lpfileid="tickdata1"
mp_filename = mp_symbol_primary + "_" + lpfileid + ".csv"
# End Params

d1 = CMqldatasetup()
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
print("cols tick2:", mv_ticks2.columns)

mv_ticks3 = d1.run_shift_data1(mv_ticks2, mp_seconds, mp_unit)

print(tabulate(mv_ticks1.head(10), showindex=False, headers=mv_ticks1.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks2.head(10), showindex=False, headers=mv_ticks2.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks3.head(10), showindex=False, headers=mv_ticks3.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))

print("mv_ticks3: ", len(mv_ticks3))
print("mv_ticks3:close ", len(mv_ticks3[['close']]))

if 'target' in mv_ticks3.columns:
    print("mv_ticks3:target ", len(mv_ticks3[['target']]))
else:
    print("mv_ticks3: 'target' column not found")

m1 = CMqlmlsetup()

# Start Params
mp_test_size = 0.2
mp_shuffle = False

# End Params
mv_ticks3=pd.DataFrame(mv_ticks3)
print("Tick3 pd before split:",mv_ticks3.head(5))
mv_X_train,mv_X_test,mv_y_train,mv_y_test = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle)

mv_X_train_scaled = m1.dl_train_model_scaled(mv_X_train)
mv_X_test_scaled = m1.dl_test_model_scaled(mv_X_test)

# Print shapes after scaling
print("mv_X_train_scaled shape:", mv_X_train_scaled.shape)
print("mv_X_test_scaled shape:", mv_X_test_scaled.shape)


# +-------------------------------------------------------------------
# Pre-tune a neural network model
# +-------------------------------------------------------------------
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
mp_act3='sigmoid'
mp_metric = 'accuracy'
mp_loss1 = 'mean_squared_error'
mp_loss2 = 'binary_crossentropy'
mp_objective='val_accuracy'

mp_cells1=128
mp_cells2=256
mp_cells3=128
mp_cells4 = 64
mp_cells5 = 1

mp_min = 32
mp_max = 128
mp_step = 2
mp_hmin = 8
mp_hmax = 128
mp_hstep = 2
mp_imin = 64
mp_imax = 256
mp_istep = 64
mp_jmin = 0.2  # Corrected value
mp_jmax = 0.5  # Corrected value
mp_jstep = 0.1

mp_validation_split = 0.2
mp_epochs = 1
mp_batch_size = 16
mp_num_trials = 1
mp_num_models=1
mp_arraysize = 100000 #560 = 100 set the size for the dense tensor array
# End Params


# Run the tuner
# Start Params
mp_objective='val_accuracy'
mp_max_trials=10  # Number of hyperparameter sets to try
mp_executions_per_trial=1  # Number of models to build and evaluate for each trial
mp_modeldatapath = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\PythonLib\tsModelData"

mp_directory=mp_modeldatapath + '\\' + 'tshybrid_ensemble_tuning_prod'
mp_project_name=mp_modeldatapath + '\\' + 'tshybrid_ensemble_model_prod'
if mp_test == True:
        mp_directory=mp_modeldatapath + '\\' + 'tshybrid_ensemble_tuning_test'
        mp_project_name=mp_modeldatapath + '\\' + 'tshybrid_ensemble_model_test' 

print("mp_directory",mp_directory)     
print("mp_project_name",mp_project_name)


# End Params


 # Truncate 'x' to match 'y'
mv_X_train_scaled = mv_X_train_scaled[:len(mv_y_train)] 
mv_X_test_scaled = mv_X_test_scaled[:len(mv_y_test)] 
mp_train_input_shape = (mv_X_train_scaled.shape)
mp_test_input_shape = (mv_X_test_scaled.shape)


############################################
# Start Test Load Data
############################################

if mp_test == True:
    mv_X_train_scaled=np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
    mv_X_test_scaled=np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature

    mv_y_train = np.random.randint(2, size=(1000,))  # Binary target
    mv_y_test = np.random.randint(2, size=(1000,))  # Binary target
    
    mp_train_input_shape = (100, 1)
    mp_test_input_shape = (100, 1)
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
mt = CMdtuner(mp_train_input_shape, mv_X_train_scaled, mv_y_train, mp_objective, mp_max_trials, mp_executions_per_trial, mp_directory, mp_project_name, mp_validation_split, mp_epochs, mp_batch_size,mp_arraysize)

best_model = mt.run_tuner()

# Print the summary of the best model
best_model.summary()


# Plot the model
#import graphviz , pydot
#from tensorflow.keras.utils import plot_model
#modelView = plot_model(best_model, to_file='SummaryModel.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


# Create a list of exactly 4 identical tensors
mv_X_train_list = [mv_X_train_scaled] * 4
# Correct the call to best_model.fit
mv_model = best_model.fit(mv_X_train_list, mv_y_train, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)

# Ensure test data is in the correct shape
mv_X_test_scaled = mv_X_test_scaled.reshape((-1, 100, 1))  # Reshape to match (100, 1)

# Create a list of exactly 4 identical tensors
mv_X_test_list = [mv_X_test_scaled] * 4

# Predict the model
predictions = pd.DataFrame(best_model.predict(mv_X_test_list))

print("Predictions:", predictions.head(5))

# Model performance
accuracy, precision, recall, f1 = m1.model_performance(best_model, mv_X_test_list, mv_y_test)
