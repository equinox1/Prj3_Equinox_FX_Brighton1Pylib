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
# packages dependencies for this module
#
import MetaTrader5 as mt5
#+-------------------------------------------------------------------
# classes for mql
#+-------------------------------------------------------------------

#--------------------------------------------------------------------
# create class  "CMqlinit"
# usage: connect services mql api
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinit:
    def __init__(self,lp_path,lp_login,lp_password,lp_server,lp_timeout,lp_portable):
        self.lp_path  =  lp_path
        self.lp_login = lp_login
        self.lp_password = lp_password
        self.lp_server = lp_server
        self.lp_timeout = lp_timeout
        self.lp_portable = lp_portable

    def __str__(self):
        return f"{self.lp_path}"
        return f"{self.lp_login}"
        return f"{self.lp_password}"
        return f"{self.lp_server}"
        return f"{self.lp_timeout}"
        return f"{self.lp_portable}"

#--------------------------------------------------------------------
# create method  "set_mql_login()".
# class: cmqlinit      
# usage: login
# \param cmqlinit    var                          
#--------------------------------------------------------------------
    def set_mql_login(lp_path,lp_login,lp_password,lp_server,lp_timeout,lp_portable):
        if mt5.initialize(path=lp_path,login=lp_login,password=lp_password,server=lp_server,timeout=lp_timeout,portable=lp_portable):
            print("Platform mt5 launched correctly")
            lp_return=mt5.last_error()
        else:
            print(f"there has been a problem with initialization: {mt5.last_error()}")
            lp_return=mt5.last_error()
    
    def get_mql_login(l):
            lp_return=mt5.last_error()  
                
#--------------------------------------------------------------------
# create class  "CMqlinitdemo"
# usage: connect services mql api
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitdemo(CMqlinit):
    pass

#--------------------------------------------------------------------
# create method  "set_mql_login()".
# class: cmqlinitdemo      
# usage: login
# \param cmqlinit    var                          
#--------------------------------------------------------------------     
        lp_Obj_New=CMqlinitdemo
        lp_path  =   r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
        lp_login = 51698985
        lp_password =  r'lsor31tz$r8aih'
        lp_server = r"icmarketssc-demo"
        lp_timeout =  60000
        lp_portable = True
        #run the setter
        lp_Obj_New.set_mql_login(lp_path,lp_login,lp_password,lp_server,lp_timeout,lp_portable)

#--------------------------------------------------------------------
# create class  "CMmqlinitdemo"
# usage: connect services mql api
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitprod(CMqlinit):
    pass