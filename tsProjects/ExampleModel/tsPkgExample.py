
"""
import tsExample 
from tsExample import Geeks

mark = Geeks() 
mark.age = 19
print(mark.age) 

mark.age = 21
print(mark.age) 
"""

from tsMqlConnect import CMqlinitdemo
lp_path  =   r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
lp_login = 51698985
lp_password =  r'lsor31tz$r8aih'
lp_server = r"icmarketssc-demo"
lp_timeout =  60000
lp_portable = True

conn2 = CMqlinitdemo

conn2.run_mql_login