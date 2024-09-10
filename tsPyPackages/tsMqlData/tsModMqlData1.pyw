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
    def __init__(self,lp_dfinit,lp_dfnew,lp_dfmql ,lp_future = 10, lp_seconds = 60,lp_step = 1 ,lp_train = 0.7,lp_random_state = 42):
        self.lp_dfinit  =  lp_dfinit
        self.lp_dfnew  =  lp_dfnew
        self.lp_future = lp_future
        self.lp_seconds = lp_seconds
        self.lp_dfmql = lp_dfmql
        self.lp_step = lp_step
        self.lp_train = lp_train
        self.lp_random_state = lp_random_state
    
#--------------------------------------------------------------------
# create method  "litmus_test()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def litmus_test():
        a=10
        b=20
        c= a + b 
        print("sum: ",c)   
        return c 

#--------------------------------------------------------------------
# create method  "get/set_mql_load_step1()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def set_mql_load_step1(lp_df,lp_dfinit,lp_dfnew ,lp_future = 10):
        # using close prices for the next i bar
        target = pd.concat([lp_dfinit['close'].shift(-i) for i in range(1, lp_future + 1)], axis=1) 
        # naming the columns
        target.columns = [f'target_close_{i}' for i in range(1, lp_future + 1)] 
        lp_dfnew = pd.dataframe({'open': lp_df['open'],'high': lp_df['high'],'low':  lp_df['low'],'close': lp_df['close'] })
    
    def get_mql_load_step1():    
        return self.lp_dfnew

#--------------------------------------------------------------------
# create method  "get/set_mql_load_step2()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def set_mql_load_step2(lp_dfnew,lp_future = 10):
        target_columns =  self.set_initial_target_step1(lp_dfnew, lp_future).dropna()
        combined_df = pd.concat([lp_dfnew, target_columns], axis=1) #concatenating the new pandas dataframe with the target columns
        combined_df = combined_df.dropna() #droping rows with nan values caused by shifting values
        target_cols_names = [f'target_close_{i}' for i in range(1, lp_future + 1)]
        x = combined_df.drop(columns=target_cols_names).values #dropping all target columns from the x array
        y = combined_df[target_cols_names].values # creating the target variables
        print(f"x={x.shape} y={y.shape}")
        combined_df.head(10)
    
    def get_mql_load_step2():
        return combined_df

#--------------------------------------------------------------------
# create method  "get/set_mql_load_step3()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------   
    def set_mql_load_step3(lp_dfmql, lp_step, lp_train_size=0.7, lp_random_state=42):
        # since we are using the open high low close ohlc values only
        lp_dfmql["next signal"] = lp_dfmql["signal"].shift(-lp_step) # the target variable from next n future values
        lp_dfmql = lp_dfmql.dropna()
        y = lp_dfmql["next signal"]
        x = lp_dfmql.drop(columns=["signal", "next signal"])
        step3result=train_test_split(x, y, train_size=lp_train_size, random_state=lp_random_state) 
    
    def get_mql_load_step3():
        return step3result

#--------------------------------------------------------------------
# create method  "get/set_load_from_mql()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------   
    def set_load_from_mql(lp_rates1 = "rates",lp_year = 2024, lp_month = 1, lp_day =1 ,lp_timezone = "etc/utc",lp_symbol = "eurusd"):
        lp_rates1= pd.dataframe()
        lp_rates1 = lp_rates1.drop(index=lp_rates1.index)
        # set time zone to utc
        timezone = pytz.timezone(lp_timezone)
        # create 'datetime' object in utc time zone to avoid the implementation of a local time zone offset
        utc_from = datetime(lp_year, lp_month, lp_day, tzinfo=lp_timezone)
        # request 100 000 eurusd ticks starting from lp_year, lp_month, lp_day in utc time zone
        lp_rates1 = mt5.copy_ticks_from(lp_symbol, utc_from, 1000000000, mt5.copy_ticks_all)
        print("ticks received:",len(lp_rates1))
    
    def get_load_from_mql():
        return self.lp_rates1

#--------------------------------------------------------------------
# create method  "get/set_load_from_mql_toframe()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def set_load_from_mql_toframe(lp_rates1 = "rates" ,lp_rates2 = "rates_frame" ,lp_year = 2024, lp_month = 1, lp_day =1 ,lp_timezone = "etc/utc",lp_symbol = "eurusd"):
        # copy rates into a new panda dataframe
        lp_rates2=pd.dataframe()
        lp_rates2 = lp_rates2.drop(index=lp_rates2.index)
        lp_rates2 = pd.dataframe(lp_rates1)
        #add and divide the bid and ask values in the data by two. this way, we obtain the average value, which will be our input for deep learning.
        lp_rates2['time']=pd.to_datetime(lp_rates2['time'], unit='s')
        lp_rates2['close']=(lp_rates2['ask']+lp_rates2['bid'])/2
    
    def get_load_from_mql_toframe():
        return lp_rates2

#--------------------------------------------------------------------
# create method  "get/set_data_frame_shift()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#-------------------------------------------------------------------- 
    def set_data_frame_shift(lp_seconds = 60, lp_rates2 = "rates_frame"):
        #shifting a dataframe in time series forecasting, is commonly done to create sequences of input and target variables
        #number_of_rows is a variable representing the number of seconds.
        #empty_rows creates a new dataframe with nan values, having the same columns as the original dataframe ( rates_frame).
        number_of_rows=lp_seconds
        empty_rows = pd.dataframe(np.nan, index=range(number_of_rows), columns=lp_rates2.columns)
        #if.append(empty_rows, ignore_index=true) appends the empty rows to the original dataframe ( df ) while ignoring the index to ensure a continuous index.
        lp_rates2= lp_rates2._append(empty_rows, ignore_index=true)
        #df['target'] = df['close'].shift(-seconds) creates a new column 'target' containing the 'close' values shifted by a negative value of the specified number of seconds. this is commonly done when preparing time series data for predictive modeling.
        lp_rates2['target'] = lp_rates2['close'].shift(-lp_seconds)
        print("rates_frame modified",lp_rates2)

    def set_data_frame_shift(lp_seconds = 60, lp_rates2 = "rates_frame"):
        return lp_rates2

#--------------------------------------------------------------------
# create method  "get/set_data_frame_shift_clean()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#-------------------------------------------------------------------- 
    def set_data_frame_shift_clean(lp_rates2):
        #now, all that's left is to clean the data so that we can use it as input with tensorflow.
        lp_rates2=lp_rates2.dropna()
        #the result is a modified dataframe ( df ) with additional rows filled with nan values and a new 'target' column for time-shifted 'close' values.
    
    def get_data_frame_shift_clean(lp_rates2):
        return lp_rates2