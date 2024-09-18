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
lp_seconds = 60
lp_unit = 's' 
mv_ticks3 = d1.run_shift_data1(mv_ticks2, lp_seconds,lp_unit)
print(tabulate(mv_ticks3, showindex=False, headers=mv_ticks1.columns,tablefmt="pretty",numalign="left",stralign="left",floatfmt=".4f"))

#+-------------------------------------------------------------------
## Build a neural network model
#+-------------------------------------------------------------------  

m1 = CMqlmlsetup
mp_k_reg = 0.01
mp_test_size = 0.2
mp_shuffle = False
mv_X_train =[]
mv_y_train =[]

mv_X_train=m1.dl_split_data_sets(mv_ticks3,mp_shuffle,1)
mv_y_train=m1.dl_split_data_sets(mv_ticks3,mp_shuffle,2)
mv_X_test=m1.dl_split_data_sets(mv_ticks3,mp_shuffle,3)
mv_y_test=m1.dl_split_data_sets(mv_ticks3,mp_shuffle,4)

"""
from sklearn.model_selection import train_test_split
X = []
y = []
X = mv_ticks3[['close']]   
y = mv_ticks3['target']

X_train=[]
X_test=[]
y_train=[]
y_test=[]
X_train = train_test_split(X, y, test_size=0.2, shuffle=False)
X_test= train_test_split(X, y, test_size=0.2, shuffle=False)
y_train = train_test_split(X, y, test_size=0.2, shuffle=False)
y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


X_train_scaled=[]
X_test_scaled=[]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

"""

mv_X_train_scaled=m1.dl_train_model_scaled(mv_X_train)
mv_X_test_scaled=m1.dl_test_model_scaled(mv_X_test)

mv_model=m1.dl_build_neuro_network(mp_k_reg,mv_X_train)
mv_model_compiled=m1.dl_compile_neuro_network(mv_model)