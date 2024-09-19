#+------------------------------------------------------------------+
#|                                                 neuropredict2.py |
#|                                                    tony shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# import standard python packages
#+-------------------------------------------------------------------
import sys
from datetime import datetime
import pytz
import jinja2
import tabulate

import pandas as pd
#numpy is widely used for numerical operations in python.
import numpy as np
#+-------------------------------------------------------------------
# import mql packages
#+-------------------------------------------------------------------
import MetaTrader5 as mt5

#======================================================
# import local packages
#======================================================

import tsMqlConnect
import tsMqlData
import tsMqlML

import keyring as kr 

# User can use the alias if they want 

from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup 
from tsMqlML import CMqlmlsetup
mv_debug = 0

#======================================================
# Start Login MQL Terminal
#======================================================
# Add to hash Vault keyring externally via CLI

cred = kr.get_credential("xercesdemo","")
c1 = CMqlinitdemo
#start Params
c1.lp_path=r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
c1.lp_login=int(cred.username)
c1.lp_password=str(cred.password)
c1.lp_server=r"ICMarketsSC-Demo"
c1.lp_timeout=60000
c1.lp_portable= True
#End Params

#=================
# Login Metatrader
#======================================================
c1.run_mql_login(c1.lp_path,c1.lp_login,c1.lp_password,c1.lp_server,c1.lp_timeout,c1.lp_portable)
#======================================================
# End Login MQL Terminal
#======================================================

#+-------------------------------------------------------------------
# Import Data from MQL
#+-------------------------------------------------------------------   
#start Params
mp_symbol_primary = "EURUSD"
mp_symbol_secondary = "GBPUSD"
mp_year = 2024
mp_month = 1
mp_day = 1
mp_timezone = "UTC"
mp_rows = 10000
mp_command = mt5.COPY_TICKS_ALL
mp_dfName="df_rates"
#End Params

d1=CMqldatasetup
mv_utc_from = d1.set_mql_timezone(mp_year,mp_month,mp_day,mp_timezone)
print("Timezone Set to : ",mv_utc_from)
mv_ticks1= pd.DataFrame(d1.run_load_from_mql(mv_debug,mp_dfName,mv_utc_from,mp_symbol_primary,mp_rows,mp_command))

#+-------------------------------------------------------------------
# Prepare Data
#+-------------------------------------------------------------------
mv_ticks2=pd.DataFrame() # Empty DataFrame
mv_ticks2 = mv_ticks2.drop(index=mv_ticks2.index)
mv_ticks2 = pd.DataFrame(mv_ticks1)

#+-------------------------------------------------------------------
# Shift Data
#+-------------------------------------------------------------------  
# Tabulate formatting
# Use seaborn to set the style
from tabulate import tabulate
#start Params
lp_seconds = 60
lp_unit = 's' 
#End Params

mv_ticks3 = d1.run_shift_data1(mv_ticks2, lp_seconds,lp_unit)
print(tabulate(mv_ticks3, showindex=False, headers=mv_ticks1.columns,tablefmt="pretty",numalign="left",stralign="left",floatfmt=".4f"))

m1 = CMqlmlsetup

#Start Params
mp_test_size = 0.2
mp_shuffle = False
 
#End Params
mv_X_train =[]
mv_y_train =[]

mv_X_train=m1.dl_split_data_sets(mv_ticks3,mp_test_size,mp_shuffle,1)
mv_y_train=m1.dl_split_data_sets(mv_ticks3,mp_test_size,mp_shuffle,2)

mv_X_test=m1.dl_split_data_sets(mv_ticks3,mp_test_size,mp_shuffle,3)
mv_y_test=m1.dl_split_data_sets(mv_ticks3,mp_test_size,mp_shuffle,4)

mv_X_train_scaled=m1.dl_train_model_scaled(mv_X_train)
mv_X_test_scaled=m1.dl_test_model_scaled(mv_X_test)



#+-------------------------------------------------------------------
# Build a neural network model
#+------------------------------------------------------------------- 
#start Params
mp_k_reg = 0.001
mp_optimizer='adam'
mp_loss='mean_squared_error'
# End Params

mv_model=m1.dl_build_neuro_network(mp_k_reg,mv_X_train,mp_optimizer,mp_loss)


"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
k_reg=0.001
model=None
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(mv_X_train.shape[1],), kernel_regularizer=l2(k_reg)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

"""
#+--------------------------------------------------------------------
# Train the model
#+--------------------------------------------------------------------
#start Params
mp_epoch=5
mp_batch_size=256
mp_validation_split=0.2
mp_verbose=1
#End Params
mv_model=m1.dl_train_model(mv_model,mv_X_train_scaled,mv_y_train,mp_epoch,mp_batch_size, mp_validation_split ,mp_verbose )

#+--------------------------------------------------------------------
# Predict the model
#+--------------------------------------------------------------------
#start Params
#mp_seconds=60
#End Params
#mv_predictions=m1.dl_predict_values(mv_ticks3,mv_model,mp_seconds)

# Print actual and predicted values for the next  n  instances
#print("Actual Value for the Last Instances:")
#print(mv_ticks2.tail(1)['close'].values)

#print("\nPredicted Value for the Next Instances:")
#print(mv_predictions[:, 0])
#df_predictions=pd.DataFrame(mv_predictions)