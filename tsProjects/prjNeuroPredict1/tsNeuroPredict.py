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
mp_rows = 10000
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
mp_seconds = 60
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


mv_X_train = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle, 1)
mv_y_train = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle, 2)

mv_X_test = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle, 3)
mv_y_test = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle, 4)

mv_X_train_scaled = m1.dl_train_model_scaled(mv_X_train)
mv_X_test_scaled = m1.dl_test_model_scaled(mv_X_test)


# +-------------------------------------------------------------------
# Build a neural network model
# +-------------------------------------------------------------------
# start Params
mp_k_reg = 0.001
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
# End Params
mv_model = m1.dl_build_neuro_network(mp_k_reg, mv_X_train, mp_optimizer, mp_loss)
# +--------------------------------------------------------------------
# Train the model
# +--------------------------------------------------------------------
# start Params
mp_epoch = 100
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
mp_seconds=60
# End Params
#print("MvTick3Tail:",mv_ticks3.tail(mp_seconds)['close'].values)
mv_predictions=m1.dl_predict_values(mv_ticks3,mv_model,mp_seconds)


# Print actual and predicted values for the next  n  instances
print("Actual Value for the Last Instances:")
print(mv_ticks2.tail(1)['close'].values)

print("\nPredicted Value for the Next Instances:")
print(mv_predictions[:, 0])
df_predictions=pd.DataFrame(mv_predictions)

# modelperformance
m1.dl_model_performance(mv_model,mv_X_train_scaled, mv_X_test_scaled)