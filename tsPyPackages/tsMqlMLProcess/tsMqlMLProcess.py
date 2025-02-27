#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqkMLProcess.py
File: tsPyPackages/tsMqkMLProcess/tsMqkMLProcess.py
Description: Load and add files and data parameters.
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

import logging
# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlMLParams import CMqlEnvMLParams, CEnvCore
from tsMqlMLTunerParams import CMqlEnvMLTunerParams

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info("Running on: %s, loadmql state: %s", os_platform, loadmql)

class CDMLProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        if loadmql:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        else:
            self.mt5 = None
        
        # Set environment parameters
        self.set_envmgr_params(kwargs)
        # Set machine learning parameters
        self.set_ml_params(kwargs)
        # Set global parameters
        self._set_global_parameters(kwargs)

    def set_envmgr_params(self, kwargs):
        """Extract environment parameters."""
        # Extract parameter sections
        env = CMqlEnvMgr() 
        self.env = env
        self.params = self.env.all_params()
        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

    def set_ml_params(self,kwargs):
        """Extract environment parameters."""  
        # Load machine learning parameters
        self.mlconfig = CMqlEnvMLParams()
        self.FEATURES_PARAMS = self.mlconfig.get_features_params()
        self.WINDOW_PARAMS = self.mlconfig.get_window_params()
        self.DEFAULT_PARAMS = self.mlconfig.get_default_params()
        logger.info("Features parameters: %s", self.FEATURES_PARAMS)
        logger.info("Window parameters: %s", self.WINDOW_PARAMS)
        logger.info("Default parameters: %s", self.DEFAULT_PARAMS)

        #load ML tune parameters 
        ml_tune_config = CMqlEnvMLTunerParams()
        self.TUNER_DEFAULT_PARAMS = ml_tune_config.get_default_params()
        logger.info("Tuner default parameters: %s", self.TUNER_DEFAULT_PARAMS)  

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        # Set the default base parameters
        self.timeval = kwargs.get('timeval', 1)
        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)

        logger.info("timeval: %d, colwidth: %d, hrows: %d", self.timeval, self.colwidth, self.hrows)

        # Set the default Feature parameters
        self.mp_ml_input_keyfeat = self.FEATURES_PARAMS["mp_ml_input_keyfeat"] = self.ml_params.get('mp_ml_input_keyfeat', 'KeyFeature')
        self.mp_ml_input_keyfeat_scaled = self.FEATURES_PARAMS["mp_ml_input_keyfeat_scaled"] = self.ml_params.get('mp_ml_input_keyfeat_scaled', 'KeyFeature_Scaled')
        self.mp_ml_output_label = self.FEATURES_PARAMS["mp_ml_output_label"] = self.ml_params.get('mp_ml_output_label', 'Label')
        self.mp_ml_output_label_scaled = self.FEATURES_PARAMS["mp_ml_output_label_scaled"] = self.ml_params.get('mp_ml_output_label_scaled', 'Label_Scaled')
        logger.info("mp_ml_input_keyfeat: %s, mp_ml_input_keyfeat_scaled: %s, mp_ml_output_label: %s, mp_ml_output_label_scaled: %s", self.mp_ml_input_keyfeat, self.mp_ml_input_keyfeat_scaled, self.mp_ml_output_label, self.mp_ml_output_label_scaled)
        # Set the default Window parameters
        self.mp_ml_cfg_period = self.WINDOW_PARAMS["mp_ml_cfg_period"] = self.ml_params.get('mp_ml_cfg_period', 24)
        self.mp_ml_cfg_period1 = self.WINDOW_PARAMS["mp_ml_cfg_period1"] = self.ml_params.get('mp_ml_cfg_period1', 24)
        self.mp_ml_cfg_period2 = self.WINDOW_PARAMS["mp_ml_cfg_period2"] = self.ml_params.get('mp_ml_cfg_period2', 6)
        self.mp_ml_cfg_period3 = self.WINDOW_PARAMS["mp_ml_cfg_period3"] = self.ml_params.get('mp_ml_cfg_period3', 1)
        self.mp_ml_tf_ma_windowin = self.WINDOW_PARAMS["mp_ml_tf_ma_windowin"] = self.ml_params.get('mp_ml_tf_ma_windowin', 'mp_ml_cfg_period')
        self.pasttimeperiods = self.WINDOW_PARAMS["pasttimeperiods"] = self.ml_params.get('pasttimeperiods', 'mp_ml_cfg_period')
        self.futuretimeperiods = self.WINDOW_PARAMS["futuretimeperiods"] = self.ml_params.get('futuretimeperiods', 'mp_ml_cfg_period1')
        self.predtimeperiods = self.WINDOW_PARAMS["predtimeperiods"] = self.ml_params.get('predtimeperiods', 1)
        logger.info("mp_ml_cfg_period: %d, mp_ml_cfg_period1: %d, mp_ml_cfg_period2: %d, mp_ml_cfg_period3: %d, mp_ml_tf_ma_windowin: %s, pasttimeperiods: %s, futuretimeperiods: %s, predtimeperiods: %s", self.mp_ml_cfg_period, self.mp_ml_cfg_period1, self.mp_ml_cfg_period2, self.mp_ml_cfg_period3, self.mp_ml_tf_ma_windowin, self.pasttimeperiods, self.futuretimeperiods, self.predtimeperiods)
        # Batch size from tuning parameters
        self.mp_ml_batch_size = self.DEFAULT_PARAMS["batch_size"] = self.mltune_params.get('batch_size', 8)
        logger.info("mp_ml_batch_size: %d", self.mp_ml_batch_size)

    def create_ml_window(self, timeval):
        """Select the data file based on the DataFrame name."""  
        features_count = len(self.mp_ml_input_keyfeat)  # Number of features in input
        labels_count = len(self.mp_ml_output_label)  # Number of labels in output
        batch_size = self.mp_ml_batch_size  # Batch size for training
        past_width = int(self.pasttimeperiods) * timeval
        future_width = int(self.futuretimeperiods) * timeval
        pred_width = int(self.predtimeperiods) * timeval

        logger.info("past_width: %d, future_width: %d, pred_width: %d, timeval: %d", past_width, future_width, pred_width, timeval)
        logger.info("features_count: %d, labels_count: %d, batch_size: %d", features_count, labels_count, batch_size)

        return past_width, future_width, pred_width, features_count, labels_count, batch_size
