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
            "pasttimeperiods": 24,
            "futuretimeperiods": 24,
            "predtimeperiods": 1,
        }

        self.FEATURES_PARAMS = {
            "mp_ml_input_keyfeat": {"FEAT1": "Close"},
            "mp_ml_input_keyfeat_scaled": {"FEAT1_SCALED": "Close_Scaled"},
            "mp_ml_output_label": {"LABEL1": "Label"},
            "mp_ml_output_label_scaled": {"LABEL1_SCALED": "Label_Scaled"},
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
            "mp_ml_rownumber": True,
            "mp_ml_remove_zeros": True,
            "mp_ml_last_col": True,
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
