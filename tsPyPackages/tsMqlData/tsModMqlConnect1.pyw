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
# Create Class  "CMqlInit"
# Usage: Connect services MQL Api
#
# Section:params
# \param  double sVar;              -  Value
#-------------------------------------------------------------------- 
class CMqlInit:
 def __init__(self,lpPath,lpLogin,lpPassword,lpServer,lpTimeout,lpPortable):
    self.path  =  lpPath
    self.login = lpLogin
    self.password = lpPassword
    self.server = lpServer
    self.timeout = lpTimeout
    self.portable = lpPortable
   
 def __str__(self):
    return f"{self.path}"
    return f"{self.login}"
    return f"{self.password}"
    return f"{self.server}"
    return f"{self.timeout}"
    return f"{self.portable}"

#--------------------------------------------------------------------
# Create Method  "Set_Mql_Login()".
# Class: CMqlInit      
# Usage: Login
# \param CMqlInit    Var                          
#--------------------------------------------------------------------
 def Set_Mql_Login(lpPath,lpLogin,lpPassword,lpServer,lpTimeout,lpPortable):
        if mt5.initialize(path=lpPath,login=lpLogin,password=lpPassword,server=lpServer,timeout=lpTimeout,portable=lpPortable):
            print("Platform MT5 launched correctly")
            lpReturn=mt5.last_error()
        else:
            print(f"There has been a problem with initialization: {mt5.last_error()}")
            lpReturn=mt5.last_error()
    
 def Get_Mql_Login(l):
       lpReturn=mt5.last_error()  
                
#--------------------------------------------------------------------
# Create Class  "CMqlInitDemo"
# Usage: Connect services MQL Api
#
# Section:params
# \param  double sVar;              -  Value
#-------------------------------------------------------------------- 
class CMqlInitDemo(CMqlInit):
     pass
 
#--------------------------------------------------------------------
# Create Method  "Set_Mql_Login()".
# Class: CMqlInitDemo      
# Usage: Login
# \param CMqlInit    Var                          
#--------------------------------------------------------------------     
lpNew=CMqlInitDemo
lpPath  =   r"C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\MQL5\Brokers\ICMarkets\terminal64.exe"
lpLogin = 51698985
lpPassword =  r'LsOR31tz$r8AIH'
lpServer = r"ICMarketsSC-Demo"
lpTimeout =  60000
lpPortable = True
#Run the setter
lpNew.Set_Mql_Login(lpPath,lpLogin,lpPassword,lpServer,lpTimeout,lpPortable)

#--------------------------------------------------------------------
# Create Class  "CMqlInitDemo"
# Usage: Connect services MQL Api
#
# Section:params
# \param  double sVar;              -  Value
#-------------------------------------------------------------------- 
class CMqlInitProd(CMqlInit):
    pass