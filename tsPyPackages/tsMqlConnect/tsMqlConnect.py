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
        self._lp_path  =  lp_path
        self._lp_login = lp_login
        self._lp_password = lp_password
        self._lp_server = lp_server
        self._lp_timeout = lp_timeout
        self._lp_portable = lp_portable
    
    #Setter and Getter lp_path  
    @property
    def lp_path(self):
        return self.lp_path

    @lp_path.setter
    def lp_path(self, value):
        self._lp_path = value
        
    #Setter and Getter lp_login  
    @property
    def lp_login(self):
        return self.lp_login
    
    @lp_login.setter
    def lp_login(self, value):
        self._lp_login = value
        
    #Setter and Getter lp_password
    @property
    def lp_password(self):
        return self.lp_password
    
    @lp_password.setter
    def lp_password(self, value):
        self._lp_password = value
        
    #Setter and Getter lp_server
    @property
    def lp_server(self):
        return self.lp_server
    
    @lp_server.setter
    def lp_server(self,value):
        self._lp_server = value    
        
    #Setter and Getter lp_timeout
    @property
    def lp_timeout(self):
        return self.lp_timeout
    
    @lp_timeout.setter
    def lp_timeout(self,value):
        self._lp_timeout = value       
        
    #Setter and Getter lp_portable
    @property
    def lp_portable(self):
        return self.lp_portable
    
    @lp_portable.setter
    def lp_portable(self,value):
        self._lp_portable = value         
        
        
    
 
#--------------------------------------------------------------------
# create method  "run_mql_login()".
# class: cmqlinit      
# usage: login
# \param cmqlinit    var                          
#--------------------------------------------------------------------
    def run_mql_login(lp_path,lp_login,lp_password,lp_server,lp_timeout,lp_portable):
        if mt5.initialize(path=lp_path,login=lp_login,password=lp_password,server=lp_server,timeout=lp_timeout,portable=lp_portable):
            print("Platform mt5 launched correctly")
            lp_return=mt5.last_error()
        else:
            print(f"there has been a problem with initialization: {mt5.last_error()}")
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
# create method  "run_mql_login()".
# class: cmqlinitdemo      
# usage: login
# \param cmqlinit    var                          
#--------------------------------------------------------------------       

    def __init__(self,lp_path,lp_login,lp_password,lp_server,lp_timeout,lp_portable):
        self._lp_path  =  lp_path
        self._lp_login = lp_login
        self._lp_password = lp_password
        self._lp_server = lp_server
        self._lp_timeout = lp_timeout
        self._lp_portable = lp_portable
    
#--------------------------------------------------------------------
# create class  "CMmqlinitdemo"
# usage: connect services mql api
#
# section:params
# \param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqlinitprod(CMqlinit):
    pass