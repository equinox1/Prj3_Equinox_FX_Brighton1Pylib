#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

import os
import sys
import logging
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLTunerParams import CMqlEnvMLTunerParams  # Ensure this exists

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info("Running on: %s, loadmql state: %s", os_platform, loadmql)

class CDMLProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
   
        self.env = CMqlEnvMgr()
        self.mlconfig = CMqlEnvMLParams()  # Initialize ML parameters
        self.ml_tune_config = CMqlEnvMLTunerParams()  # Initialize ML tuner parameters
        
        # Initialize MetaTrader5 safely
        self.mt5 = None
        if loadmql:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
                if not self.mt5.initialize():
                    logger.error("Failed to initialize MetaTrader5: %s", self.mt5.last_error())
            except ImportError:
                logger.warning("MetaTrader5 module not found. Continuing without it.")

        # Load all configurations
        self._set_envmgr_params(kwargs)
        self._set_ml_params(kwargs)
        self._set_global_parameters(kwargs)

    def _set_envmgr_params(self, kwargs):
        """Extract environment parameters safely."""
        try:
            self.params = self.env.all_params()
        except Exception as e:
            logger.error("Failed to initialize CMqlEnvMgr: %s", e)
            self.params = {}

        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

    def _set_ml_params(self, kwargs):
        """Extract machine learning parameters safely."""
        try:
            self.FEATURES_PARAMS = self.mlconfig.get_features_params()
            self.WINDOW_PARAMS = self.mlconfig.get_window_params()
            self.DEFAULT_PARAMS = self.mlconfig.get_default_params()
            self.TUNER_DEFAULT_PARAMS = self.ml_tune_config.get_default_params()

            logger.info("Features parameters: %s", self.FEATURES_PARAMS)
            logger.info("Window parameters: %s", self.WINDOW_PARAMS)
            logger.info("Default parameters: %s", self.DEFAULT_PARAMS)
            logger.info("Tuner default parameters: %s", self.TUNER_DEFAULT_PARAMS)

        except Exception as e:
            logger.error("Error loading ML parameters: %s", e)
            self.FEATURES_PARAMS, self.WINDOW_PARAMS, self.DEFAULT_PARAMS, self.TUNER_DEFAULT_PARAMS = {}, {}, {}, {}

    def _set_global_parameters(self, kwargs):
        """Set global parameters from environment or user input."""
        self.timeval = kwargs.get('timeval', 1)
        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)

        logger.info("timeval: %d, colwidth: %d, hrows: %d", self.timeval, self.colwidth, self.hrows)

        # Set default parameters
        self.mp_ml_cfg_period = self.WINDOW_PARAMS.get("mp_ml_cfg_period", 24)
        self.mp_ml_batch_size = self.mltune_params.get('batch_size', 8)

        logger.info("Batch size: %d", self.mp_ml_batch_size)

    def create_ml_window(self, timeval):
        """Select the data file based on the DataFrame name."""
        features_count = len(self.FEATURES_PARAMS.get("mp_ml_input_keyfeat", {}))
        labels_count = len(self.FEATURES_PARAMS.get("mp_ml_output_label", {}))
        batch_size = self.mp_ml_batch_size

        past_width = int(self.WINDOW_PARAMS.get("pasttimeperiods", 24)) * timeval
        future_width = int(self.WINDOW_PARAMS.get("futuretimeperiods", 24)) * timeval
        pred_width = int(self.WINDOW_PARAMS.get("predtimeperiods", 1)) * timeval

        logger.info("past_width: %d, future_width: %d, pred_width: %d, timeval: %d", past_width, future_width, pred_width, timeval)
        logger.info("features_count: %d, labels_count: %d, batch_size: %d", features_count, labels_count, batch_size)

        return past_width, future_width, pred_width, features_count, labels_count, batch_size
