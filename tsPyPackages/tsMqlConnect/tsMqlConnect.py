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
# \param  double svar;              -  value
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
# \param cmqlinit    var                          
#--------------------------------------------------------------------
    def run_mql_login(self):
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
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitdemo(CMqlinit):
    pass

#-------------------------------------------------------------------
# create class  "CMmqlinitdemo"
# usage: connect services mql api
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitprod(CMqlinit):
    pass