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

import pandas as pd
#numpy is widely used for numerical operations in python.
import numpy as np
#+-------------------------------------------------------------------
# import mql packages
#+-------------------------------------------------------------------
import MetaTrader5 as mt5

#+-------------------------------------------------------------------
# import ai packages scikit learns
#+-------------------------------------------------------------------
#sklearn library. it includes utilities for data preprocessing, model evaluation, and model selection.
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from scikit-learn, which is often used for cross-validation in machine learning.
from sklearn.model_selection import KFold

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2

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

#======================================================
# Login MQL Terminal
#======================================================
# Add to hash Vault keyring externally via CLI

cred = kr.get_credential("xercesdemo","")

print("Username : ",cred.username) 
print("Password : ",cred.password)

c1 = CMqlinitdemo

c1.lp_path=r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
c1.lp_login=int(cred.username)
c1.lp_password=str(cred.password)
c1.lp_server=r"ICMarketsSC-Demo"
c1.lp_timeout=60000
c1.lp_portable= True

print("lp_path:",c1.lp_path)
print("lp_login:",c1.lp_login)
print("lp_password:",c1.lp_password)
print("lp_server:",c1.lp_server)
print("lp_timeout:",c1.lp_timeout)
print("lp_portable:",c1.lp_portable)

#lp_path=r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
#mt5.initialize(path=lp_path,login=51698985,password=r"lsor31tz$r8aih",server=r"ICMarketsSC-Demo",timeout=60000,portable=True)

c1.run_mql_login(c1.lp_path,c1.lp_login,c1.lp_password,c1.lp_server,c1.lp_timeout,c1.lp_portable)

"""
#+-------------------------------------------------------------------
# run Data Setup steps
#+-------------------------------------------------------------------
mp_future = 10
mv_new_df=set_mql_newdf_step(mp_new_df)
mv_target_columns = set_mql_target_step(mv_new_df,mp_future).dropna()cls
mv combined_df = pd.concat(mv_new_df,mv_target_columns,axis =1)#concatenating the new pandas dataframe with the target
mv combined_df = mv combined_df.dropna() #dropping rows with nan values caused by shifting values
mv_target_cols_names = [f'target_close_{i}' for i in range(1, mp_future + 1)]
mv_x = mv_combined_df.drop(columns=mv_target_cols_names).values #dropping all target columns from the x array
mv_y = mv_combined_df[mv_target_cols_names].values # creating the target variables
print(f"mv_x={mv_x.shape} mv_y={mv_y.shape}")
mv_combined_df.head(10)

#+-------------------------------------------------------------------
# Import Data from MQL
#+-------------------------------------------------------------------   
mv_loaded_df=set_load_from_mql()
#+-------------------------------------------------------------------
# Prepare Training data
#+-------------------------------------------------------------------
"""
