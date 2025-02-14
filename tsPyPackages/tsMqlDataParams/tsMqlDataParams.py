#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
import logging
# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")


import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import warnings
import posixpath  # For path handling
import pandas as pd
import numpy as np

class CMqlEnvData:
    def __init__(self,globalenv, **kwargs):
        self.kwargs = kwargs
        self.globalenv = globalenv
         
        # Initialize the MT5 api
        if loadmql:
            import MetaTrader5 as mt5
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        else:
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', None)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)

        # key switching parameters
        self.mp_data_cfg_usedata = kwargs.get('mp_data_cfg_usedata', 'loadfilerates')
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', True)
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', True)
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', True)
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', True)
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', 'df_rates1')
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', 'df_rates2')
        self.mp_data_rows = kwargs.get('mp_data_rows', 1000)
        self.mp_data_rowcount = kwargs.get('mp_data_rowcount', 10000)

        self.mp_data_filename1 = kwargs.get('mp_data_filename1', None)
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', None)
      
  
        self.mp_data_path = self.globalenv.get_params().get('mp_data_path', None)
       
        # Load configuration parameters with default values
        self.mp_data_data_label = kwargs.get('mp_data_data_label', 3)
        self.mp_data_history_size = kwargs.get('mp_data_history_size', 5)
        self.mp_data_timeframe = kwargs.get('mp_data_timeframe', None)
        self.mp_data_tab_rows = kwargs.get('mp_data_tab_rows', 10)
        self.mp_data_tab_width = kwargs.get('mp_data_tab_width', 30)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)
        self.mp_data_show_head = kwargs.get('mp_data_show_head', False)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)

        
       
    def get_params(self):
        return self.__dict__  # Returns all attributes as a dictionary