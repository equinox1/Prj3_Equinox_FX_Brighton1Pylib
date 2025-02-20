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

# Each parameter Class now extends the CEnvCore class and has a 
# DEFAULT_PARAMS dictionary that contains the default values for the parameters.
# The CEnvCore class has a get_params method that returns the parameters
# dictionary and a set_param method that allows you to set a parameter dynamically.
# The CMqlEnvDataParams and CMqlEnvMLParams classes have the DEFAULT_PARAMS dictionary 
# that contains the default values for the parameters. The CMqlEnvDataParams class has
# parameters related to data loading, while the CMqlEnvMLParams class has parameters 
# related to machine learning.

from tsMqlEnvCore import CEnvCore

class CMqlEnvMLParams(CEnvCore):
   """Manage machine learning tuner parameters."""
   def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization

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