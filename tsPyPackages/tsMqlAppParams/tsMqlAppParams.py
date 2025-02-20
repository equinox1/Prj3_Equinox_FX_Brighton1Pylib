"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlAppParams.py
File: tsPyPackages/tsMqlAppParams/tsMqlAppParams.py
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

from tsMqlEnvCore import CEnvCore

class CMqlEnvAppParams(CEnvCore):
   def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization
        self.file_setter("EURUSD", "tickdata1.csv", "ratesdata1.csv")

   def file_setter(self, symbol, datafilename1, datafilename2):
        self.set("mp_symbol_primary", symbol)  # Fix: Use 'set' instead of 'set_param'
        self.set("MPDATAFILE1", symbol + "_" + datafilename1) 
        self.set("MPDATAFILE2", symbol + "_" + datafilename2)
        return self.all_params()  # Fix: Use 'all_params()' instead of 'get_params()'

   DEFAULT_PARAMS = {
        "broker": "METAQUOTES",  
        "mp_symbol_secondary": "EURCHF", 
        "DFNAME1": "df_rates1",
        "DFNAME2": "df_rates2",
        "mp_data_cfg_usedata": 'loadfilerates',
        "mp_data_rows": 2000,
        "mp_data_rowcount": 10000,
        "ONNX_save": False,
        "mp_ml_show_plot": False,
        "mp_ml_hard_run": True,
        "mp_ml_tunemode": True,
        "mp_ml_tunemodeepochs": True,
        "mp_ml_Keras_tuner": 'hyperband',
        "batch_size": 4,
        "all_modelscale": 2,
        "cnn_modelscale": 2,
        "lstm_modelscale": 2,
        "gru_modelscale": 2,
        "trans_modelscale": 2,
        "transh_modelscale": 1,
        "transff_modelscale": 4,
        "dense_modelscale": 2
    }
