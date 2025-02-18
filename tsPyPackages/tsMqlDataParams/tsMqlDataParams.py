"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlDataParams.py
File: tsPyPackages/tsMqlDataParams/tsMqlDataParams.py
Description: Load adat files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

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

import logging

logger = logging.getLogger(__name__)

def get_global_param(all_params, param_name, default=None):
    """Safely fetch global environment parameters."""
    if hasattr(all_params, 'get_params') and callable(all_params.get_params):
        return all_params.get_params().get(param_name, default)
    logger.warning(f"Global parameter '{param_name}' is not available.")
    return default

class CMqlEnvDataParams:
    DEFAULTS = {
        'mp_data_data_label': 3,
        'mp_data_history_size': 5,
        'mp_data_timeframe': None,
        'mp_data_tab_rows': 10,
        'mp_data_tab_width': 30,
        'mp_data_rownumber': False,
        'mp_data_show_dtype': False,
        'mp_data_show_head': False,
        'mp_data_command_ticks': None,
        'mp_data_command_rates': None,
        'mp_data_cfg_usedata': 'loadfilerates',
        'mp_data_loadapiticks': True,
        'mp_data_loadapirates': True,
        'mp_data_loadfileticks': True,
        'mp_data_loadfilerates': True,
        'mv_data_dfname1': 'df_rates1',
        'mv_data_dfname2': 'df_rates2',
        'mp_data_rows': 1000,
        'mp_data_rowcount': 10000,
        'mp_data_filename1': None,
        'mp_data_filename2': None,
    }

    def __init__(self, **kwargs):
        self.params = self.DEFAULTS.copy()
        self.params.update(kwargs)
        
        # Handle dynamic values
        if 'mp_data_command_ticks' not in kwargs:
            self.params['mp_data_command_ticks'] = mt5.COPY_TICKS_ALL if loadmql else None
        
    def get_params(self):
        """Return all configuration parameters as a dictionary."""
        return self.params.copy()

    def set_param(self, key, value):
        """Allows overriding parameters dynamically."""
        if key in self.params:
            self.params[key] = value
        else:
            logger.warning(f"Parameter '{key}' is not recognized.")
