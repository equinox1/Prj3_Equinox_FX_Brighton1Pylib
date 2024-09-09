#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Packages dependencies for this module
#
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------

#--------------------------------------------------------------------
# Create Class  "CMqlDataSetup"
# Usage: MQL data services
#
# Section:params
# \param  double sVar;              -  Value
#-------------------------------------------------------------------- 
class CMqlDataSetup:
 def __init__(self,lpDfinit,lpDfNew,lpDfMql ,lpFuture = 10, lpSeconds = 60,lpStep = 1 ,lpTrain = 0.7,lpRandom_State = 42):
    self.lpDfinit  =  lpDfinit
    self.lpDfNew  =  lpDfNew
    self.lpFuture = lpFuture
    self.lpSeconds = lpSeconds
    self.lpDfMql = lpDfMql
    self.lpStep = lpStep
    self.lpTrain = lpTrain
    self.lpRandom_State = lpRandom_State
    
#--------------------------------------------------------------------
# Create Method  "LitmusTest()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------     
 def LitmusTest():
     a=10
     b=20
     c= a + b 
     print("Sum: ",c)   
     return c 
 
 #--------------------------------------------------------------------
# Create Method  "Get/Set_Mql_Load_Step1()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------     
 def Set_Mql_Load_Step1(lpDf,lpDfinit,lpDfNew ,lpFuture = 10):
    # using close prices for the next i bar
    target = pd.concat([lpDfinit['Close'].shift(-i) for i in range(1, lpFuture + 1)], axis=1) 
    # naming the columns
    target.columns = [f'target_close_{i}' for i in range(1, lpFuture + 1)] 
    lpDfNew = pd.DataFrame({'Open': lpDf['Open'],'High': lpDf['High'],'Low':  lpDf['Low'],'Close': lpDf['Close'] })
    
 def Get_Mql_Load_Step1():    
     return lpDfNew
  
#--------------------------------------------------------------------
# Create Method  "Get/Set_Mql_Load_Step2()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------     
 def Set_Mql_Load_Step2(lpDfNew,lpFuture = 10):
    target_columns =  self.Set_Initial_Target_Step1(lpDfNew, lpFuture).dropna()
    combined_df = pd.concat([lpDfNew, target_columns], axis=1) #concatenating the new pandas dataframe with the target columns
    combined_df = combined_df.dropna() #droping rows with NaN values caused by shifting values
    target_cols_names = [f'target_close_{i}' for i in range(1, lpFuture + 1)]
    X = combined_df.drop(columns=target_cols_names).values #dropping all target columns from the x array
    y = combined_df[target_cols_names].values # creating the target variables
    print(f"x={X.shape} y={y.shape}")
    combined_df.head(10)
    
 def Get_Mql_Load_Step2():
     return combined_df

#--------------------------------------------------------------------
# Create Method  "Get/Set_Mql_Load_Step3()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------   
 def Set_Mql_Load_Step3(lpDfMql, lpStep, lpTrain_size=0.7, lpRandom_state=42):
    # Since we are using the Open High Low Close OHLC values only
    lpDfMql["next signal"] = lpDfMql["Signal"].shift(-lpStep) # The target variable from next n future values
    lpDfMql = lpDfMql.dropna()
    y = lpDfMql["next signal"]
    X = lpDfMql.drop(columns=["Signal", "next signal"])
    Step3Result=train_test_split(X, y, train_size=lpTrain_size, random_state=lpRandom_state) 
    
 def Get_Mql_Load_Step3():
    return Step3Result

#--------------------------------------------------------------------
# Create Method  "Get/Set_Load_From_Mql()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------   
 def Set_Load_From_Mql(lpRates1 = "rates",lpYear = 2024, lpMonth = 1, lpDay =1 ,lpTimezone = "Etc/UTC",lpSymbol = "EURUSD"):
    lpRates1= pd.DataFrame()
    lpRates1 = lpRates1.drop(index=lpRates1.index)
    # set time zone to UTC
    timezone = pytz.timezone(lpTimezone)
    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(lpYear, lpMonth, lpDay, tzinfo=lpTimezone)
    # request 100 000 EURUSD ticks starting from lpYear, lpMonth, lpDay in UTC time zone
    lpRates1 = mt5.copy_ticks_from(lpSymbol, utc_from, 1000000000, mt5.COPY_TICKS_ALL)
    print("Ticks received:",len(lpRates1))
    
def Get_Load_From_Mql():
    return lpRates1
 
#--------------------------------------------------------------------
# Create Method  "Get/Set_Load_From_Mql_toFrame()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#--------------------------------------------------------------------     
def Set_Load_From_Mql_toFrame(lpRates1 = "rates" ,lpRates2 = "rates_frame" ,lpYear = 2024, lpMonth = 1, lpDay =1 ,lpTimezone = "Etc/UTC",lpSymbol = "EURUSD"):
    # Copy rates into a new panda dataframe
    lpRates2=pd.DataFrame()
    lpRates2 = lpRates2.drop(index=lpRates2.index)
    lpRates2 = pd.DataFrame(lpRates1)
    #add and divide the bid and ask values in the data by two. This way, we obtain the average value, which will be our input for deep learning.
    lpRates2['time']=pd.to_datetime(lpRates2['time'], unit='s')
    lpRates2['close']=(lpRates2['ask']+lpRates2['bid'])/2
    
def Get_Load_From_Mql_toFrame():
    return lpRates2
 
#--------------------------------------------------------------------
# Create Method  "Get/Set_Data_Frame_Shift()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#-------------------------------------------------------------------- 
def Set_Data_Frame_Shift(lpSeconds = 60, lpRates2 = "rates_frame"):
    #Shifting a DataFrame in time series forecasting, is commonly done to create sequences of input and target variables
    #number_of_rows is a variable representing the number of seconds.
    #empty_rows creates a new DataFrame with NaN values, having the same columns as the original DataFrame ( rates_frame).
    number_of_rows=lpSeconds
    empty_rows = pd.DataFrame(np.nan, index=range(number_of_rows), columns=lpRates2.columns)
    #if.append(empty_rows, ignore_index=True) appends the empty rows to the original DataFrame ( df ) while ignoring the index to ensure a continuous index.
    lpRates2= lpRates2._append(empty_rows, ignore_index=True)
    #df['target'] = df['close'].shift(-seconds) creates a new column 'target' containing the 'close' values shifted by a negative value of the specified number of seconds. This is commonly done when preparing time series data for predictive modeling.
    lpRates2['target'] = lpRates2['close'].shift(-lpSeconds)
    print("rates_frame modified",lpRates2)
 
def Set_Data_Frame_Shift(lpSeconds = 60, lpRates2 = "rates_frame"):
     return lpRates2

#--------------------------------------------------------------------
# Create Method  "Get/Set_Data_Frame_Shift_Clean()".
# Class: CMqlDataSetup      
# Usage: MQL Data
# \param  Var                          
#-------------------------------------------------------------------- 
def Set_Data_Frame_Shift_Clean(lpRates2):
    #Now, all that's left is to clean the data so that we can use it as input with TensorFlow.
    lpRates2=lpRates2.dropna()
    #The result is a modified DataFrame ( df ) with additional rows filled with NaN values and a new 'target' column for time-shifted 'close' values.
    
def Get_Data_Frame_Shift_Clean(lpRates2):
     return lpRates2