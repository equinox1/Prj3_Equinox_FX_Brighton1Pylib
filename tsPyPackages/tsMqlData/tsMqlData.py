#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
"""
Basic conventions
=================
The PEP 8 Style Guide provides guidelines for naming conventions, indentation, and spacing. Here are some of the most important basic conventions:

Naming conventions:
==================
Variables, functions, and methods should be named in lowercase with underscores between words (snake_case).
Class names should be written in CamelCase (starting with a capital letter).
Constants should be written in all capital letters with underscores between words.

Indentation:
====================
Use 4 spaces for indentation.
Don’t use tabs.

Spacing:
====================
Use spaces around operators and after commas.
Don’t use spaces inside parentheses, brackets, or braces.
Whitespace
===========
Whitespace is an important aspect of code readability.

The PEP 8 Style Guide recommends using whitespace to make code more readable. 

Use a single blank line to separate logical sections of code.
Don’t use multiple blank lines in a row.
Use whitespace around operators, but don’t use too much.
Use whitespace to align code, but don’t use too much.
"""
# packages dependencies for this module
#

import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import numpy as np

#+-------------------------------------------------------------------
# import ai packages scikit learns
#+-------------------------------------------------------------------
#sklearn library. it includes utilities for data preprocessing, model evaluation, and model
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from scikit-learn, which is often used for cross-validat
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
#+-------------------------------------------------------------------
# classes for mql
#+-------------------------------------------------------------------

#--------------------------------------------------------------------
# create class  "CMqldatasetup"
# usage: mql data services
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqldatasetup:
    def __init__(self,lp_target ,lp_dfinit,lp_dfnew,lp_dfmql ,lp_future = 10, lp_seconds = 60,lp_step = 1 ,lp_train = 0.7,lp_random_state = 42):
        self._lp_dfinit  =  lp_dfinit
        self._lp_dfnew  =  lp_dfnew
        self._lp_target = lp_target
        self._lp_future = lp_future
        self._lp_seconds = lp_seconds
        self._lp_dfmql = lp_dfmql
        self._lp_step = lp_step
        self._lp_train = lp_train
        self._lp_random_state = lp_random_state
        
        #Setter and Getter lp_dfinit  
        @property
        def lp_dfinit(self):
            return self.lp_dfinit
        
        @lp_dfinit.setter
        def lp_dfinit(self, value):
            self.lp_dfinit = value
        
        #Setter and Getter lp_dfnew 
        @property
        def lp_dfnew(self):
            return self.lp_dfnew
        
        @lp_dfnew.setter
        def lp_dfnew(self, value):
            self.lp_dfnew = value

        #Setter and Getter lp_target
        @property
        def lp_target(self):
            return self.lp_target
        
        @lp_target.setter
        def lp_target(self, value):
            self.lp_target = value
        
        #Setter and Getter lp_future
        @property
        def lp_future(self):
            return self.lp_future
        
        @lp_future.setter
        def lp_future(self, value):
            self.lp_future = value       
        
        #Setter and Getter lp_seconds
        @property
        def lp_seconds(self):
            return self.lp_seconds
        
        @lp_seconds.setter
        def lp_seconds(self, value):
            self.lp_seconds = value          
        
        #Setter and Getter lp_dfmql
        @property
        def lp_dfmql(self):
            return self.lp_dfmql
        
        @lp_dfmql.setter
        def lp_dfmql(self,value):
            self.lp_dfmql = value                 
        
        #Setter and Getter lp_dfmql
        @property
        def lp_dfmql(self):
            return self.lp_dfmql
        
        @lp_dfmql.setter
        def lp_dfmql(self,value):
            self.lp_dfmql = value    
            
        #Setter and Getter lp_step
        @property
        def lp_step(self):
            return self.lp_step
        
        @lp_step.setter
        def lp_step(self,value):
            self.lp_step = value                       
            
        #Setter and Getter lp_train
        @property
        def lp_train(self):
            return self.lp_train
        
        @lp_train.setter
        def lp_train(self,value):
            self.lp_train = value                
            
        #Setter and Getter lp_random_state
        @property
        def lp_random_state(self):
            return self.lp_random_
        
        @lp_train.setter
        def lp_random_state(self,value):
            self.lp_random_state = value   

# create method  "setmql_timezone()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          

    def set_mql_timezone(lp_year=2024, lp_month=1, lp_day=1, lp_timezone="etc/UTC"):
        lv_timezone = pytz.timezone(lp_timezone)  # Set the timezone
        native_dt = datetime(lp_year, lp_month, lp_day)  # Create a native datetime object
        return lv_timezone.localize(native_dt)

#--------------------------------------------------------------------
# create method  "run_load_from_mql()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------         
        
    def run_load_from_mql(self,lp_debug,lp_rates1 = "rates",lp_utc_from = "2023-01-01 00:00:00+00:00",lp_symbol = "JPYUSD", lp_rows = 1000,lp_flag = mt5.COPY_TICKS_ALL ):
        # request 100 000 eurusd ticks starting from lp_year, lp_month, lp_day in utc time zone
        if lp_debug == 1:
            print("lp_rates1:",lp_rates1)
            print("lp_utc_from:",lp_utc_from)
            print("lp_symbol:",lp_symbol)
            print("lp_rows:",lp_rows)
            print("lp_flag:",lp_flag)
        lp_rates1= pd.DataFrame()
        lp_rates1 = lp_rates1.drop(index=lp_rates1.index)
        lp_rates1 = mt5.copy_ticks_from(lp_symbol, lp_utc_from, lp_rows, lp_flag )
        print("ticks received:",len(lp_rates1))
        return lp_rates1
    
#--------------------------------------------------------------------
# create method  "run_shift_data1()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------         
    def run_shift_data1(self,lp_df = [], lp_seconds = 60 , lp_unit = 's'):
        lv_seconds = lp_seconds
        lv_number_of_rows = lv_seconds
        lp_df.style.set_properties(**{'text-align': 'left'})
        lp_df['time'] = pd.to_datetime(lp_df['time'], unit=lp_unit)
        lp_df['close'] = (lp_df['ask'] + lp_df['bid']) / 2
        lv_empty_rows = pd.DataFrame(np.nan, index=range(lv_number_of_rows), columns=lp_df.columns)
        lp_df = lp_df._append(lv_empty_rows, ignore_index=True)
        lp_df['target'] = lp_df['close'].shift(-lv_seconds)
        lp_df = lp_df.dropna()
        lp_df.style.set_properties(**{'text-align': 'left'})
        print("lpDf",lp_df.tail(10))
        return lp_df