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
            'mp_data_load': False,
            'mp_data_save': False,
            'mp_data_savefile': False,
            'mp_data_rownumber': True,
            'mp_data_data_label': 3,
            'mp_data_history_size': 5,
            'mp_data_timeframe': 'mt5.TIMEFRAME_H4',
            'mp_data_tab_rows': 10,
            'mp_data_tab_width': 30,
            'mp_data_rownumber': False,
            'mp_data_show_dtype': False,
            'mp_data_show_head': False,
            'mp_data_command_ticks': None,
            'mp_data_command_rates':None,
            'mp_data_cfg_usedata': 'loadfilerates',
            'mp_data_loadapiticks': True,
            'mp_data_loadapirates': True,
            'mp_data_loadfileticks': True,
            'mp_data_loadfilerates': True,
            'mv_data_dfname1': 'df_rates1',
            'mv_data_dfname2': 'df_rates2',
            'mp_data_rows': 1000,
            'mp_data_rowcount': 10000,
            'mp_data_filename1':  'tickdata1',
            'mp_data_filename2': 'ratesdata1',
            'df1_mp_data_filter_int' : False,
            'df1_mp_data_filter_flt' : False,
            'df1_mp_data_filter_obj' : False,
            'df1_mp_data_filter_dtmi' : False,
            'df1_mp_data_filter_dtmf' : False,
            'df1_mp_data_dropna' : False,
            'df1_mp_data_merge' : False,
            'df1_mp_data_convert' : False,
            'df1_mp_data_drop' : False,
            'df2_mp_data_filter_int' : False,
            'df2_mp_data_filter_flt' : False,
            'df2_mp_data_filter_obj' : False,
            'df2_mp_data_filter_dtmi' : False,
            'df2_mp_data_filter_dtmf' : False,
            'df2_mp_data_dropna' : False,
            'df2_mp_data_merge' : False,
            'df2_mp_data_convert' : False,
            'df2_mp_data_drop' : False,
            'df3_mp_data_filter_int' : False,
            'df3_mp_data_filter_flt' : False,
            'df3_mp_data_filter_obj' : False,
            'df3_mp_data_filter_dtmi' : False,
            'df3_mp_data_filter_dtmf' : False,
            'df3_mp_data_dropna' : False,
            'df3_mp_data_merge' : False,
            'df3_mp_data_convert' : False,
            'df3_mp_data_drop' : False,
            'df4_mp_data_filter_int' : False,
            'df4_mp_data_filter_flt' : False,
            'df4_mp_data_filter_obj' : False,
            'df4_mp_data_filter_dtmi' : False,
            'df4_mp_data_filter_dtmf' : False,
            'df4_mp_data_dropna' : False,
            'df4_mp_data_merge' : False,
            'df4_mp_data_convert' : False,
            'df4_mp_data_drop' : False,
         }
   
