import logging
import os
import warnings
import posixpath  # For path handling
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

# Initialize logger
logger = logging.getLogger("tsMqlDataParams")
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger as ts_logger, config

# Run platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# Conditionally import MetaTrader5
if loadmql:
    import MetaTrader5 as mt5


class CMqlEnvData:
    def __init__(self, globalenv, **kwargs):
        self.globalenv = globalenv
        self.kwargs = kwargs

        # Initialize MT5 API if available
        self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL if loadmql else None)
        self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        
        # Key switching parameters
        self.mp_data_cfg_usedata = kwargs.get('mp_data_cfg_usedata', 'loadfilerates')
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', True)
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', True)
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', True)
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', True)
        
        # Data storage parameters
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', 'df_rates1')
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', 'df_rates2')
        self.mp_data_rows = kwargs.get('mp_data_rows', 1000)
        self.mp_data_rowcount = kwargs.get('mp_data_rowcount', 10000)

        # File paths
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', None)
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', None)
        self.mp_data_path = self._get_global_param('mp_data_path')
        
        # Additional configuration parameters
        self.mp_data_data_label = kwargs.get('mp_data_data_label', 3)
        self.mp_data_history_size = kwargs.get('mp_data_history_size', 5)
        self.mp_data_timeframe = kwargs.get('mp_data_timeframe', None)
        self.mp_data_tab_rows = kwargs.get('mp_data_tab_rows', 10)
        self.mp_data_tab_width = kwargs.get('mp_data_tab_width', 30)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)
        self.mp_data_show_head = kwargs.get('mp_data_show_head', False)

    def _get_global_param(self, param_name, default=None):
        """Safely fetch global environment parameters."""
        if hasattr(self.globalenv, 'get_params') and callable(self.globalenv.get_params):
            return self.globalenv.get_params().get(param_name, default)
        logger.warning(f"Global parameter '{param_name}' is not available.")
        return default
    
    def get_params(self):
        """Return all configuration parameters as a dictionary."""
        return self.__dict__
