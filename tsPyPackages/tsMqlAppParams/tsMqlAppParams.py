"""
Filename: tsMqlAppParams.py
File: tsPyPackages/tsMqlAppParams/tsMqlAppParams.py
Description: Load adat files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1 (Optimized)
License: (Optional) e.g., MIT License
"""

from tsMqlEnvCore import CEnvCore

class CMqlEnvAppParams(CEnvCore):
    DEFAULT_PARAMS = {
        "mp_app_primary_symbol": "EURUSD",  # Corrected typo
        "mp_app_secondary_symbol": "EURCHF",
        "mp_app_broker": "METAQUOTES",
        "mp_app_server": "",
        "mp_app_timeout": 60000,
        "mp_app_portable": True,
        "mp_app_env": "demo",
        "mp_app_dfname1": "df_rates1",
        "mp_app_dfname2": "df_rates2",
        "mp_app_cfg_usedata": 'df_file_rates', #  df_api_ticks, df_api_rates, df_file_ticks, df_file_rates
        "mp_app_rows": 2000,
        "mp_app_rowcount": 10000,
        "mp_app_ONNX_save": False,
        "mp_app_ml_show_plot": False,
        "mp_app_ml_hard_run": True,
        "mp_app_ml_tunemode": True,
        "mp_app_ml_tunemodeepochs": True,
        "mp_app_ml_Keras_tuner": 'hyperband',
        "mp_app_ml_batch_size": 4,
        "mp_app_ml_all_modelscale": 2,
        "mp_app_ml_cnn_modelscale": 2,
        "mp_app_ml_lstm_modelscale": 2,
        "mp_app_ml_gru_modelscale": 2,
        "mp_app_trans_modelscale": 2,
        "mp_app_transh_modelscale": 1,
        "mp_app_transff_modelscale": 4,
        "mp_app_dense_modelscale": 2
    }

    def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)
        self.set_data_filenames("EURUSD", "tickdata1.csv", "ratesdata1.csv") # More descriptive name

    def set_data_filenames(self, symbol, datafilename1, datafilename2):
        self.set("mp_symbol_primary", symbol)
        self.set("MPDATAFILE1", f"{symbol}_{datafilename1}")  # f-strings for cleaner formatting
        self.set("MPDATAFILE2", f"{symbol}_{datafilename2}")

