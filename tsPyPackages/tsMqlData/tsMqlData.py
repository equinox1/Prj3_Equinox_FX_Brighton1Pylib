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
#--------------------------------------------------------------------   
    def set_mql_timezone(lp_year = 2024, lp_month = 1, lp_day =1 ,lp_timezone = "UTC"):
        lv_timezone = pytz.timezone(lp_timezone) # set time zone to utc
        # create 'datetime' object in utc time zone to avoid the implementation of a local time zone offset
        lv_utc_from = datetime(lp_year, lp_month, lp_day, tzinfo=lv_timezone)
        return lv_utc_from

#--------------------------------------------------------------------
# create method  "run_load_from_mql()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------         
        
    def run_load_from_mql(lp_rates1 = "rates",lp_utc_from = "2023-01-01 00:00:00+00:00",lp_symbol = "JPYUSD", lp_rows = 1000,lp_flag = mt5.COPY_TICKS_ALL ):
        # request 100 000 eurusd ticks starting from lp_year, lp_month, lp_day in utc time zone
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
# create method  "set_load_from_mql_toframe()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def set_load_from_mql_toframe(lp_rates1 = "rates" ,lp_rates2 = "rates_frame"):
        # copy rates into a new panda dataframe
        lv_rates2=pd.dataframe()
        lv_rates2 = lv_rates2.drop(index=lp_rates2.index)
        lv_rates2 = pd.dataframe(lp_rates1)
        #add and divide the bid and ask values in the data by two. this way, we obtain the average value, which will be our input for deep learning.
        lv_rates2['time']=pd.to_datetime(lv_rates2['time'], unit='s')
        lv_rates2['close']=(lv_rates2['ask']+lv_rates2['bid'])/2
        return lv_rates2

#-------------------------------------------------------
# create method  "set_data_frame_shift()".
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
        lp_rates2= lp_rates2._append(empty_rows, ignore_index=True)
        #df['target'] = df['close'].shift(-seconds) creates a new column 'target' containing the 'close' values shifted by a negative value of the specified number of seconds. this is commonly done when preparing time series data for predictive modeling.
        lp_rates2['target'] = lp_rates2['close'].shift(-lp_seconds)
        print("rates_frame modified",lp_rates2)
        return lp_rates2
#--------------------------------------------------------------------
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#-------------------------------------------------------------------- 
    def set_data_frame_shift_clean(lp_rates2):
        #now, all that's left is to clean the data so that we can use it as input with tensorflow.
        lp_rates2=lp_rates2.dropna()
        #the result is a modified dataframe ( df ) with additional rows filled with nan values and 
        # a new 'target' column for time-shifted 'close' values.
        return lp_rates2
#--------------------------------------------------------------------
# create method  "set_mql_newdf_step()".
# class: cmqldatasetup      
# usage: mql data
# \param  var                          
#--------------------------------------------------------------------     
    def set_mql_newdf_step(lp_target) :
        lv_new_df = pd.DataFrame({'Open': lp_target['Open'],'High': lp_target['High'],'Low': lp_target['Low'],'Close': lp_target['Close'] })
        return lv_new_df     
#--------------------------------------------------------------------   
# create method  "set_mql_target_step()".
# class: cmqldatasetup      
# usage: mql data
# \param  var 
#--------------------------------------------------------------------                         
    def set_mql_target_step(lp_df,lp_future = 10):
        lv_target = pd.concat([lp_df['close'].shift(-i) for i in range(1, lp_future + 1)], axis=1) # using close prices for the next i
        lv_target.columns = [f'target_close_{i}' for i in range(1, lp_future + 1)] # naming the columns
        return lv_target  
#--------------------------------------------------------------------     
# usage: This function creates the new target variable using the column "Signal" from the dataset.
# usage: The target variable is taken from the index step+1 value in the signal column
# usage:  At step 1, the next signal will be 2, at step 2,  the next signal will be 3, and so on.
# usage: This means that in step 1, we create a dataset in such a way the target variable is a value from 1
# usage: bar in the future, and so on. All the independent variables remain the same.
# \param  var                          
#--------------------------------------------------------------------   
    def multi_steps_data_process(lp_dfmql, lp_step, lp_train_size=0.7, lp_random_state=42):
        # since we are using the open high low close ohlc values only
        lp_dfmql["next signal"] = lp_dfmql["signal"].shift(-lp_step) # the target variable from next n future values
        lp_dfmql = lp_dfmql.dropna()
        y = lp_dfmql["next signal"]
        x = lp_dfmql.drop(columns=["signal", "next signal"])
        lv_result=train_test_split(x, y, train_size=lp_train_size, random_state=lp_random_state) 
        return lv_result
#-------------------------------------------------------------------- 