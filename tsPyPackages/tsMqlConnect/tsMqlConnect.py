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
    def __init__(self,path,login,password,server,timeout,portable):
        self._path  =  path
        self._login = login
        self._password = password
        self._server = server
        self._timeout = timeout
        self._portable = portable
    
    #Setter and Getter path  
    @property
    def path(self):
        return self.path

    @path.setter
    def path(self, value):
        self._path = value.encode("utf-8")
        
    #Setter and Getter login  
    @property
    def login(self):
        return self.login
    
    @login.setter
    def login(self, value):
        self._login = value
        
    #Setter and Getter password
    @property
    def password(self):
        return self.password
    
    @password.setter
    def password(self, value):
        self._password = value
        
    #Setter and Getter server
    @property
    def server(self):
        return self.server
    
    @server.setter
    def server(self,value):
        self._server = value
        
    #Setter and Getter timeout
    @property
    def timeout(self):
        return self.timeout
    
    @timeout.setter
    def timeout(self,value):
        self._timeout = value 
        
    #Setter and Getter portable
    @property
    def portable(self):
        return self.portable
    
    @portable.setter
    def portable(self,value):
        self._portable = value         
        
#--------------------------------------------------------------------
# create method  "run_mql_login()".
# class: cmqlinit      
# usage: login
# \param cmqlinit    var                          
#--------------------------------------------------------------------
    def run_mql_login(lppath,lplogin,lppassword,lpserver,lptimeout,lpportable):
        if mt5.initialize(path=lppath,login=lplogin,password=lppassword,server=lpserver,timeout=lptimeout,portable=lpportable):
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