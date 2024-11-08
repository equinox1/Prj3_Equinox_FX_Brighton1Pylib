# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    tony shepherd |
# |                                                  https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "tony shepherd"
# property link      "https://www.xercescloud.co.uk"
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

from tabulate import tabulate

from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup
from tsMqlMLTune import CMqlmltuner


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
c1.path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/brokers/icmarkets/terminal64.exe"
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
"""
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
mv_ticks1 = pd.DataFrame(
    d1.run_load_from_mql(
        mv_debug,
        mp_dfName,
        mv_utc_from,
        mp_symbol_primary,
        mp_rows,
        mp_command
    )
)
mv_ticks1.head(10)

"""
import pandas as pd

lp_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"

lpfileid="tickdata1"

lp_filename = "EURUSD" + "_" + lpfileid + ".csv"

lpmergepath = lp_path + "//" + lp_filename

print("1:lpmergepath:", lpmergepath)

# Check if file can be opened
try:
    with open(lpmergepath, 'r') as f:
        print("2:File opened successfully")
except PermissionError as e:
    print("Permission denied:", e)

# Read CSV file
try:
    df = pd.read_csv(lpmergepath)
    print(df.head())  # Preview first 5 rows
except FileNotFoundError as e:
    print("File not found:", e)
except pd.errors.ParserError as e:
    print("Error parsing the file:", e)

