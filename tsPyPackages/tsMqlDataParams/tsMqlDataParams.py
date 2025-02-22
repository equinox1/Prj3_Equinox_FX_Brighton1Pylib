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
# Each parameter Class now extends the CEnvCore class and has a 
# DEFAULT_PARAMS dictionary that contains the default values for the parameters.
# The CEnvCore class has a get_params method that returns the parameters
# dictionary and a set_param method that allows you to set a parameter dynamically.
# The CMqlEnvDataParams and CMqlEnvMLParams classes have the DEFAULT_PARAMS dictionary 
# that contains the default values for the parameters. The CMqlEnvDataParams class has
# parameters related to data loading, while the CMqlEnvMLParams class has parameters 
# related to machine learning.

from tsMqlEnvCore import CEnvCore

class CMqlEnvDataParams(CEnvCore):
   def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization

   DEFAULT_PARAMS = {
            'mp_data_data_label': 3,
            'mp_data_history_size': 5,
            'mp_data_timeframe': 'H4',
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

