#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

# classes for mql
#+-------------------------------------------------------------------
import MetaTrader5 as mt5
from MetaTrader5 import *
#--------------------------------------------------------------------
# create class  "CMqlinit"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinit:
    def __init__(self, MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV,**kwargs):
        self.MPPATH = kwargs.get('MPPATH', None)
        self.MPLOGIN = kwargs.get('MPLOGIN', None)
        self.MPPASS = kwargs.get('MPPASS', 'password')
        self.MPSERVER = kwargs.get('MPSERVER', None)
        self.MPTIMEOUT = kwargs.get('MPTIMEOUT', 60000)
        self.MPPORTABLE = kwargs.get('MPPORTABLE', True)
        self.MPENV = kwargs.get('MPENV', 'demo')
        
#--------------------------------------------------------------------
# create method  "run_mql_login()".
# class: cmqlinit      
# usage: login
# /param cmqlinit    var                          
#--------------------------------------------------------------------
    def run_mql_login(self, MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV):
        if mt5.initialize(path=MPPATH, login=MPLOGIN, password=MPPASS, server=MPSERVER, timeout=MPTIMEOUT, portable=MPPORTABLE):
            print("Platform mt5 launched correctly: Ver:", mt5.version(),"Info: ", mt5.account_info())
            print("Environment:", MPENV)
            return True
        else:
            print(f"there has been a problem with initialization: {mt5.last_error()}")
            return mt5.last_error()