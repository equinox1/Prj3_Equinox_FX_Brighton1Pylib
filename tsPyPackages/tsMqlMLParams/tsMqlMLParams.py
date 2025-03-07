#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqkMLParams.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

from tsMqlEnvCore import CEnvCore

class CMqlEnvMLParams(CEnvCore):
    """Manage machine learning tuner parameters."""

    def __init__(self, **kwargs):
        """Initialize with default parameters, allowing overrides."""
        
        self.WINDOW_PARAMS = {
            "mp_ml_cfg_period1": 24,  # Hours
            "mp_ml_cfg_period2": 6,   # Hours
            "mp_ml_cfg_period3": 1,   # Hours
            "mp_ml_cfg_period": 24,   # Hours
            "mp_ml_tf_ma_windowin": 24,  # Hours
            "mp_ml_pasttimeperiods": 24,
            "mp_ml_futuretimeperiods": 24,
            "mp_ml_predtimeperiods": 1,
            "mp_ml_lookahead_periods": 24,
        }

        self.FEATURES_PARAMS = {
            "mp_ml_input_keyfeat": "Close",
            "Feature1": "Close",
            "Feature2": "High",
            "Feature3": "Low",
            "Feature4": "Open",
            "Feature5": "Volume",
            "Feature6": "HLAvg",
            "Feature7": "SMA",
            "Feature8": "LogReturns",
            "mp_ml_input_keyfeat_scaled": "Close_Scaled",
            "Feature1_scaled": "Close_Scaled",
            "Feature2_scaled": "High_Scaled",
            "Feature3_scaled": "Low_Scaled",
            "Feature4_scaled": "Open_Scaled",
            "Feature5_scaled": "Volume_Scaled",
            "Feature6_scaled": "HLAvg_Scaled",
            "Feature7_scaled": "SMA_Scaled",
            "Feature8_scaled": "LogReturns_Scaled",
            "mp_ml_output_label": "Label",
            "mp_ml_output_label_scaled": "Label_Scaled",
            "Label1" : "Label",
            "Label1_Scaled" : "Label_Scaled",
            "Label2" : "Target",
            "Label2_Scaled" : "Target_Scaled",
            }


        self.DEFAULT_PARAMS = {
            "mp_ml_tf_shiftin": 1,
            "mp_ml_hl_avg_col": 'HLAvg',
            "mp_ml_ma_col": 'SMA',
            "mp_ml_returns_col": 'LogReturns',
            "mp_ml_returns_col_scaled": 'LogReturns_scaled',
            "mp_ml_create_label": False,
            "mp_ml_create_label_scaled": False,
            "mp_ml_run_avg": True,
            "mp_ml_run_avg_scaled": True,
            "mp_ml_run_ma": True,
            "mp_ml_run_ma_scaled": True,
            "mp_ml_run_returns": True,
            "mp_ml_run_returns_scaled": True,
            "mp_ml_run_returns_shifted": True,
            "mp_ml_run_returns_shifted_scaled": True,
            "mp_ml_run_label": True,
            "mp_ml_run_label_scaled": True,
            "mp_ml_run_label_shifted": True,
            "mp_ml_run_label_shifted_scaled": True,
            "mp_ml_log_stationary": True,
            "mp_ml_remove_zeros": True,
            "mp_ml_last_col": True,
            "mp_ml_last_col_scaled": True,
            "mp_ml_first_col": True,
            "mp_ml_dropna": True,
            "mp_ml_dropna_scaled": True,
        }

        # Merge all parameters
        merged_params = {**self.DEFAULT_PARAMS, **self.FEATURES_PARAMS, **self.WINDOW_PARAMS, **kwargs}
        super().__init__(custom_params=merged_params)

    def get_features_params(self):
        """Return feature parameters."""
        return self.FEATURES_PARAMS  

    def get_window_params(self):
        """Return window parameters."""
        return self.WINDOW_PARAMS   

    def get_default_params(self):
        """Return default parameters."""
        return self.DEFAULT_PARAMS
