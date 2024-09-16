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
mv_debug = 0

#======================================================
# Start Login MQL Terminal
#======================================================
# Add to hash Vault keyring externally via CLI

cred = kr.get_credential("xercesdemo","")
c1 = CMqlinitdemo

c1.lp_path=r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
c1.lp_login=int(cred.username)
c1.lp_password=str(cred.password)
c1.lp_server=r"ICMarketsSC-Demo"
c1.lp_timeout=60000
c1.lp_portable= True

if mv_debug == 1:
    print("lp_path:",c1.lp_path)
    print("lp_login:",c1.lp_login)
    print("lp_password:",c1.lp_password)
    print("lp_server:",c1.lp_server)
    print("lp_timeout:",c1.lp_timeout)
    print("lp_portable:",c1.lp_portable)
else:
    pass

# Login Metatrader
c1.run_mql_login(c1.lp_path,c1.lp_login,c1.lp_password,c1.lp_server,c1.lp_timeout,c1.lp_portable)
#======================================================
# End Login MQL Terminal
#======================================================

#+-------------------------------------------------------------------
# Import Data from MQL
#+-------------------------------------------------------------------   
mv_symbol_primary = "EURUSD"
mv_symbol_secondary = "GBPUSD"
mv_year = 2024
mv_month = 1
mv_day = 1
mv_timezone = "UTC"

mv_rows = 10000
mv_command = mt5.COPY_TICKS_ALL

d1=CMqldatasetup
mv_utc_from = d1.set_mql_timezone(mv_year,mv_month,mv_day,mv_timezone)
print("Timezone Set to : ",mv_utc_from)
mv_ticks1= pd.DataFrame(d1.run_load_from_mql(mv_debug,"mv_dfrates",mv_utc_from,mv_symbol_primary,mv_rows,mv_command))

# Tabulate formatting
# Use seaborn to set the style
from tabulate import tabulate
print(tabulate(mv_ticks1, showindex=False, headers=mv_ticks1.columns,tablefmt="grid",numalign="left",stralign="left",floatfmt=".4f"))
print("Count: ",mv_ticks1.count())

#+-------------------------------------------------------------------
# Prepare Data
#+-------------------------------------------------------------------  
# This code converts the 'time' column to datetime format using seconds as the unit
# and calculates the average of 'ask' and 'bid' values, assigning the result to a new
# column named 'close' in the rates_frame DataFrame.The choice of input data for deep learning
# in financial applications, such as predicting price movements, depends on various factors, and 
# there isn't a one-size-fits-all solution. However, there are some considerations and general
# guidelines to keep in mind:

mv_ticks2=pd.DataFrame()
# Empty DataFrame
mv_ticks2 = mv_ticks2.drop(index=mv_ticks2.index)
mv_ticks2 = pd.DataFrame(mv_ticks1)
#To simplify things, let's add and divide the bid and ask values in the data by two.
# This way, we obtain the average value, which will be our input for deep learning.
mv_ticks2['time']=pd.to_datetime(mv_ticks2['time'], unit='s')
mv_ticks2['close']=(mv_ticks2['ask']+mv_ticks2['bid'])/2

#+-------------------------------------------------------------------
# Shift Data
#+-------------------------------------------------------------------  
#Shifting a DataFrame in the context of deep learning, particularly in time series forecasting,
#is commonly done to create sequences of input and target variables. Here are the reasons why shifting
# is used in the context of deep learning for time series prediction:
#Temporal Dependencies: Deep learning models, such as recurrent neural networks (RNNs)
# or long short-term memory networks (LSTMs), can capture temporal dependencies in sequential data. 
# Shifting the DataFrame allows you to create sequences where each input sequence corresponds to a segment of past data
# and the corresponding target sequence represents the future data.
# Sequential Learning: Deep learning models are effective at learning patterns and dependencies in sequential data. 
# By shifting the DataFrame, you ensure that the input and target sequences align temporally, 
# allowing the model to learn from the historical context and make predictions based on that context.
# Training Input-Output Pairs: Shifting helps in creating training examples for the deep learning model. 
# Each row in the shifted DataFrame can be considered an input-output pair, where the input is 
# a sequence of past observations, and the output is the target variable to be predicted in the future.

#number_of_rows is a variable representing the number of seconds.
#empty_rows creates a new DataFrame with NaN values, having the same columns as the original DataFrame ( df ).
#df.append(empty_rows, ignore_index=True) appends the empty rows to the original DataFrame ( df ) 
#while ignoring the index to ensure a continuous index.
#df['target'] = df['close'].shift(-seconds) creates a new column 'target' containing the 'close' values 
# shifted by a negative value of the specified number of seconds. 
# This is commonly done when preparing time series data for predictive modeling.

mv_seconds =60
mv_number_of_rows = mv_seconds 

empty_rows = pd.DataFrame(np.nan, index=range(mv_number_of_rows), columns=mv_ticks2.columns)

mv_ticks2 = mv_ticks2._append(empty_rows, ignore_index=True)
mv_ticks2['target'] = mv_ticks2['close'].shift(-mv_seconds)
#The result is a modified DataFrame ( df ) with additional rows filled with NaN values and a new 'target' column for time-shifted 'close' values.
mv_ticks2=mv_ticks2.dropna()
print("======================DF Modified:=================================")
print(tabulate(mv_ticks2, showindex=False, headers=mv_ticks1.columns,tablefmt="grid",numalign="left",stralign="left",floatfmt=".4f"))
