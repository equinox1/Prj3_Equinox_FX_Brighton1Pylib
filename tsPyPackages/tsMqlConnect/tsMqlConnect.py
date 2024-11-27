#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

"""
Basic conventions
=================
The PEP 8 Style Guide provides guidelines for naming convent

Naming conventions:
==================
Variables, functions, and methods should be named in lowerca
Class names should be written in CamelCase (starting with a 
Constants should be written in all capital letters with unde

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

The PEP 8 Style Guide recommends using whitespace to make co

Use a single blank line to separate logical sections of code
Don’t use multiple blank lines in a row.
Use whitespace around operators, but don’t use too much.
Use whitespace to align code, but don’t use too much.
"""

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
    def __init__(self, lppath, lplogin, lppassword, lpserver, lptimeout, lpportable):
        self._lppath = lppath
        self._lplogin = lplogin
        self._lppassword = lppassword
        self._lpserver = lpserver
        self._lptimeout = lptimeout
        self._lpportable = lpportable

#--------------------------------------------------------------------
# create method  "run_mql_login()".
# class: cmqlinit      
# usage: login
# /param cmqlinit    var                          
#--------------------------------------------------------------------

    def run_mql_login(self, lppath, lplogin, lppassword, lpserver, lptimeout, lpportable):
        if mt5.initialize(path=self._lppath, login=self._lplogin, password=self._lppassword, server=self._lpserver, timeout=self._lptimeout, portable=self._lpportable):
            print("Platform mt5 launched correctly")
            return mt5.last_error()
        else:
            print(f"there has been a problem with initialization: {mt5.last_error()}")
            return mt5.last_error()

#--------------------------------------------------------------------
# create class  "CMqlinitdemo"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitdemo(CMqlinit):
    pass

#-------------------------------------------------------------------
# create class  "CMmqlinitdemo"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitprod(CMqlinit):
    pass

#-------------------------------------------------------------------
# create class  "CMmqliniticmdemo"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqliniticmdemo(CMqlinit):
    def __init__(self, **kwargs):
        
        # Load dataset and model configuration parameters
        self.MPBASEPATH = kwargs['MPBASEPATH'] if 'MPBASEPATH' in kwargs else r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"        
        self.MPBROKPATH = kwargs['MPBROKPATH'] if 'MPBROKPATH' in kwargs else r"brokers/icmarkets/terminal64.exe"
        self.MPPATH = kwargs['MPPATH'] if 'MPPATH' in kwargs else self.MPBASEPATH + self.MPBROKPATH     

        self.MPLOGIN = kwargs['MPLOGIN'] if 'MPLOGIN' in kwargs else "username"
        self.MPPASS = kwargs['MPPASS'] if 'MPPASS' in kwargs else "password"
        self.MPSERVER = kwargs['MPSERVER'] if 'MPSERVER' in kwargs else r"ICMarketsSC-Demo"
        self.MPTIMEOUT = kwargs['MPTIMEOUT'] if 'MPTIMEOUT' in kwargs else 60000
        self.MPPORTABLE = kwargs['MPPORTABLE'] if 'MPPORTABLE' in kwargs else True
        self.run_mql_login(self.MPPATH, self.MPLOGIN, self.MPPASS, self.MPSERVER, self.MPTIMEOUT, self.MPPORTABLE)


#-------------------------------------------------------------------
# create class  "CMmqliniticmprod"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqliniticmprod(CMqlinit):
    pass

#-------------------------------------------------------------------
# create class  "CMmqlinitmetademo"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqliniticmdemo(CMqlinit):
    pass

#-------------------------------------------------------------------
# create class  "CMmqlinitmetaprod"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitmetaprod(CMqlinit):
    pass