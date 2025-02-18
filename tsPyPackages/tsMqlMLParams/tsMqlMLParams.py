"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename : tsMqlMLParams.py
File : tsPyPackages/tsMqlMLParams/tsMqlMLParams.py
Description : Load machine learning parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""



import logging
from tsMqlPlatform import run_platform, platform_checker, logger as ts_logger, config

# Initialize logger
logger = logging.getLogger("tsMqlDataParams")
logging.basicConfig(level=logging.INFO)

# Run platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

def get_global_param(all_params, param_name, default=None):
    """Safely fetch global environment parameters."""
    if hasattr(all_params, 'get_params') and callable(all_params.get_params):
        return all_params.get_params().get(param_name, default)
    logger.warning(f"Global parameter '{param_name}' is not available.")
    return default

class CMqlEnvMLParams:
    DEFAULT_PARAMS = {
        "mp_ml_cfg_period": 24,
        "mp_ml_cfg_period1": 24,  # Hours
        "mp_ml_cfg_period2": 6,   # Hours
        "mp_ml_cfg_period3": 1,   # Hours
        "mp_ml_tf_ma_windowing": 24,
        "mp_ml_tf_shiftin": 1,
        "mp_ml_hl_avg_col": 'HLAvg',
        "mp_ml_ma_col": 'SMA',
        "mp_ml_returns_col": 'LogReturns',
        "mp_ml_returns_col_scaled": 'LogReturns_scaled',
        "mp_ml_create_label": False,
        "mp_ml_create_label_scaled": False,
        "mp_ml_input_keyfeat": {'Close'},
    }
    
    def __init__(self, **overrides):
        self.params = None
        
        # Override default parameters with provided values
        self.params = {key: overrides.get(key, get_global_param(self.params, key, default))
                       for key, default in self.DEFAULT_PARAMS.items()}
        
        # Derived parameters
        self.params["mp_ml_output_label"] = self.params["mp_ml_input_keyfeat"]
        self.params["mp_ml_input_keyfeat_scaled"] = {feat + '_Scaled' for feat in self.params["mp_ml_input_keyfeat"]}
        self.params["mp_ml_output_label_scaled"] = {targ + '_Scaled' for targ in self.params["mp_ml_output_label"]}
        self.params["mp_ml_output_label_count"] = len(self.params["mp_ml_output_label"])

    def get_params(self):
        """Returns a dictionary of all set parameters."""
        return self.params
