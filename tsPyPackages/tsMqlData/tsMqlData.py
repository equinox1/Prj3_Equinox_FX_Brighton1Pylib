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
import datetime
import pytz
from datetime import datetime, timedelta
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
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2

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
    def __init__(self,target ,dfinit,dfnew,dfmql ,future = 10, seconds = 60,step = 1 ,train = 0.7,random_state = 42):
        self._dfinit  =  dfinit
        self._dfnew  =  dfnew
        self._target = target
        self._future = future
        self._seconds = seconds
        self._dfmql = dfmql
        self._step = step
        self._train = train
        self._random_state = random_state

        #Setter and Getter dfinit
        @property
        def dfinit(self):
            return self.dfinit

        @dfinit.setter
        def dfinit(self, value):
            self.dfinit = value

        #Setter and Getter dfnew
        @property
        def dfnew(self):
            return self.dfnew

        @dfnew.setter
        def dfnew(self, value):
            self.dfnew = value

        #Setter and Getter target
        @property
        def target(self):
            return self.target

        @target.setter
        def target(self, value):
            self.target = value

        #Setter and Getter future
        @property
        def future(self):
            return self.future

        @future.setter
        def future(self, value):
            self.future = value
        #Setter and Getter seconds

        @property
        def seconds(self):
            return self.seconds

        @seconds.setter
        def seconds(self, value):
            self.seconds = value

        #Setter and Getter dfmql
        @property
        def dfmql(self):
            return self.dfmql

        @dfmql.setter
        def dfmql(self,value):
            self.dfmql = value

        #Setter and Getter dfmql
        @property
        def dfmql(self):
            return self.dfmql

        @dfmql.setter
        def dfmql(self,value):
            self.dfmql = value

        #Setter and Getter step
        @property
        def step(self):
            return self.step

        @step.setter
        def step(self,value):
                        self.step = value
        #Setter and Getter train

        @property
        def train(self):
            return self.train

        @train.setter
        def train(self,value):
            self.train = value

        #Setter and Getter random_state
        @property
        def random_state(self):
            return self.random_

        @train.setter
        def random_state(self,value):
            self.random_state = value

# create method  "setmql_timezone()".
# class: cmqldatasetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------
    def set_mql_timezone(year=2024, month=1, day=1, timezone=None):
        # Set the timezone using the provided timezone string
        tz = pytz
        tz.timezone(timezone)
        # Return a datetime object with the specified year, month, day, and timezone
        return datetime(year, month, day, tzinfo=tz)
#--------------------------------------------------------------------
# create method  "run_load_from_mql()".
# class: cmqldatasetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------
    def run_load_from_mql(self, rates1 = "rates", utc_from = "2023-01-01 00:00:00+00:00", symbol = "JPYUSD", rows = 1000, flag = mt5.COPY_TICKS_ALL):
        # request 100 000 eurusd ticks starting from year, month, day in utc time zone
        if self == 1:
            print("rates1:",rates1)
            print("utc_from:",utc_from)
            print("symbol:",symbol)
            print("rows:",rows)
            print("flag:",flag)
        rates1= pd.DataFrame()
        rates1 = rates1.drop(index=rates1.index)

        rates1 = mt5.copy_ticks_from(symbol, utc_from, rows, flag )
        print("ticks received:",len(rates1))
        return rates1

#--------------------------------------------------------------------
# create method  "run_shift_data1()".
# class: cmqldatasetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------

    def run_shift_data1(self, seconds = 60, unit = 's'):

        seconds =seconds
        number_of_rows = seconds
        #+-------------------------------------------------------------------
        # This code converts the 'time' column to datetime format using seconds as the unit
        # and calculates the average of 'ask' and 'bid' values, assigning the result to a new
        # column named 'close' in the rates_frame DataFrame. unit 's' = seconds
        #+-------------------------------------------------------------------
        # head time(datetime),bid(float5),ask(float5),last  volume(int),time_msc(datetime),flags(int),volume_real(int),close(float5),target(float5)
        self.style.set_properties(**{'text-align': 'left'})
        #df['time']=pd.to_datetime(df['time'], unit=unit)
        self['time'] = pd.to_datetime(
            self.time, format='%Y%m%d%H%M%S', errors='coerce'
        )
        #df['time_msc']= pd.to_datetime(df.time_msc, format='%Y%m%d%H%M%S', errors='coerce')

        self['close'] = (self['ask'] + self['bid']) / 2

        empty_rows = pd.DataFrame(
            np.nan, index=range(number_of_rows), columns=self.columns
        )
        self = self._append(empty_rows, ignore_index=True)
        self['target'] = self['close'].shift(-seconds)
        self = self.dropna()
        self.style.set_properties(**{'text-align': 'left'})
        return self