#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlOverrides/tsMqlOverrides.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

import sys
import platform
import os
from pathlib import Path
import yaml

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import textwrap
from tabulate import tabulate

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlEnvCore import CEnvCore

# Initialize platform checker and logger
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")


class CMqlOverrides(CEnvCore):
    """
    Environment setup for the MQL-based platform, handling paths and configurations.
    Reads configuration parameters from config.yaml.
    """

    def __init__(self, config_path="config.yaml", **kwargs):
        # Load configuration from YAML file
        self.config = self._load_config(config_path)
        # Pass any additional kwargs if needed
        super().__init__(custom_params=kwargs)
   
        self.env = CMqlEnvMgr()
        self.local_data_params = {}  
        self.params = self.env.all_params()

        self._initialize_env_mgr()
        self._set_data_overrides()
        self._set_feature_overrides()  # Ensure feature overrides are applied
        self._set_ml_overrides()
        self._set_mltune_overrides()
        self._set_app_overrides()  # Now including app-specific overrides

    def _load_config(self, config_path):
        """Load configuration parameters from a YAML file."""
        config_file = Path(config_path)
        if config_file.is_file():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                logger.info("Configuration loaded from %s", config_path)
                return config
            except Exception as e:
                logger.error("Error loading configuration from %s: %s", config_path, e)
                return {}
        else:
            logger.error("Configuration file %s not found.", config_path)
            return {}

    def _initialize_env_mgr(self):
        """Initializes the environment manager and loads parameters.
           Combines the configuration from config.yaml with existing parameters.
        """
        try:
            self.env = CMqlEnvMgr()
            self.params = self.env.all_params()
            # Merge loaded config into environment parameters (if needed)
            self.params.update(self.config)
            logger.info("Environment Manager initialized successfully.")
        except Exception as e:
            logger.critical("Failed to initialize CMqlEnvMgr: %s", e)
            self.params = {}

    def _set_data_overrides(self):
        """Overrides default data parameters using values from config.yaml if available."""
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping data overrides.")
            return

        data_config = self.config.get("data", {})
        data_overrides = {
            "mp_data_load": data_config.get("mp_data_load", True),
            "mp_data_save": data_config.get("mp_data_save", False),
            "mp_data_savefile": data_config.get("mp_data_savefile", False),
            "mp_data_rownumber": data_config.get("mp_data_rownumber", False),
            "mp_data_data_label": data_config.get("mp_data_data_label", 3),
            "mp_data_history_size": data_config.get("mp_data_history_size", 5),
            "mp_data_timeframe": data_config.get("mp_data_timeframe", "mt5.TIMEFRAME_H4"),
            "mp_data_tab_rows": data_config.get("mp_data_tab_rows", 10),
            "mp_data_tab_width": data_config.get("mp_data_tab_width", 30),
            "mp_data_show_dtype": data_config.get("mp_data_show_dtype", False),
            "mp_data_show_head": data_config.get("mp_data_show_head", False),
            "mp_data_command_ticks": data_config.get("mp_data_command_ticks", None),
            "mp_data_command_rates": data_config.get("mp_data_command_rates", None),
            "mp_data_cfg_usedata": data_config.get("mp_data_cfg_usedata", "loadfilerates"),
            "mp_data_loadapiticks": data_config.get("mp_data_loadapiticks", True),
            "mp_data_loadapirates": data_config.get("mp_data_loadapirates", True),
            "mp_data_loadfileticks": data_config.get("mp_data_loadfileticks", True),
            "mp_data_loadfilerates": data_config.get("mp_data_loadfilerates", True),
            "mp_data_filename1": data_config.get("mp_data_filename1", "tickdata1"),
            "mp_data_filename2": data_config.get("mp_data_filename2", "ratesdata1"),
            "mv_data_dfname1": data_config.get("mv_data_dfname1", "df_rates1"),
            "mv_data_dfname2": data_config.get("mv_data_dfname2", "df_rates2"),
            "mp_data_rows": data_config.get("mp_data_rows", 1000),
            "mp_data_rowcount": data_config.get("mp_data_rowcount", 10000),
         }

        try:
            self.env.override_params({"data": data_overrides})
            logger.info("Data parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override data parameters: %s", e)

    def _set_feature_overrides(self):
        """Overrides feature parameters using settings from config.yaml if provided."""
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping feature overrides.")
            return

        feat_config = self.config.get("FEATURES_PARAMS", {})
        if feat_config:
            try:
                self.env.override_params({"feat": feat_config})
                logger.info("Feature parameters overridden successfully.")
            except Exception as e:
                logger.error("Failed to override feature parameters: %s", e)
        else:
            logger.info("No feature overrides provided.")

    def _set_ml_overrides(self):
        """Overrides ML-related parameters using config.yaml settings if available."""
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping ML overrides.")
            return

        ml_config = self.config.get("ML_PARAMS", {})
        ml_overrides = {
            "mp_ml_tf_shiftin": ml_config.get("mp_ml_tf_shiftin", 1),
            "mp_ml_hl_avg_col": ml_config.get("mp_ml_hl_avg_col", "HLAvg"),
            "mp_ml_ma_col": ml_config.get("mp_ml_ma_col", "SMA"),
            "mp_ml_returns_col": ml_config.get("mp_ml_returns_col", "LogReturns"),
            "mp_ml_returns_col_scaled": ml_config.get("mp_ml_returns_col_scaled", "LogReturns_scaled"),
            "mp_ml_create_label": ml_config.get("mp_ml_create_label", True),
            "mp_ml_create_label_scaled": ml_config.get("mp_ml_create_label_scaled", False),
            "mp_ml_run_avg": ml_config.get("mp_ml_run_avg", True),
            "mp_ml_run_avg_scaled": ml_config.get("mp_ml_run_avg_scaled", True),
            "mp_ml_run_ma": ml_config.get("mp_ml_run_ma", True),
            "mp_ml_run_ma_scaled": ml_config.get("mp_ml_run_ma_scaled", True),
            "mp_ml_run_returns": ml_config.get("mp_ml_run_returns", True),
            "mp_ml_run_returns_scaled": ml_config.get("mp_ml_run_returns_scaled", True),
            "mp_ml_run_returns_shifted": ml_config.get("mp_ml_run_returns_shifted", True),
            "mp_ml_run_returns_shifted_scaled": ml_config.get("mp_ml_run_returns_shifted_scaled", True),
            "mp_ml_run_label": ml_config.get("mp_ml_run_label", True),
            "mp_ml_run_label_scaled": ml_config.get("mp_ml_run_label_scaled", True),
            "mp_ml_run_label_shifted": ml_config.get("mp_ml_run_label_shifted", True),
            "mp_ml_run_label_shifted_scaled": ml_config.get("mp_ml_run_label_shifted_scaled", True),
            "mp_ml_log_stationary": ml_config.get("mp_ml_log_stationary", True),
            "mp_ml_remove_zeros": ml_config.get("mp_ml_remove_zeros", True),
            "mp_ml_last_col": ml_config.get("mp_ml_last_col", True),
            "mp_ml_last_col_scaled": ml_config.get("mp_ml_last_col_scaled", True),
            "mp_ml_first_col": ml_config.get("mp_ml_first_col", True),
            "mp_ml_dropna": ml_config.get("mp_ml_dropna", True),
            "mp_ml_dropna_scaled": ml_config.get("mp_ml_dropna_scaled", True),
         }  
        try:
            self.env.override_params({"ml": ml_overrides})
            logger.info("ML parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override ML parameters: %s", e)

    def _set_mltune_overrides(self):
        """Overrides ML tuning parameters if provided in config.yaml."""
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping ML tuning overrides.")
            return

        mltune_config = self.config.get("ML_TUNING_PARAMS", {})
        mltune_overrides = {
            "mp_ml_tunemode": mltune_config.get("mp_ml_tunemode", True),
            "tunemodeepochs": mltune_config.get("tunemodeepochs", 100),
            "seed": mltune_config.get("seed", 42),
            "tuner_id": mltune_config.get("tuner_id", 1),
            "train_split": mltune_config.get("train_split", 0.7),
            "validation_split": mltune_config.get("validation_split", 0.2),
            "test_split": mltune_config.get("test_split", 0.1),
            "batch_size": mltune_config.get("batch_size", 8),
            "input_width": mltune_config.get("input_width", 24),
            "shift": mltune_config.get("shift", 24),
            "total_window_size": mltune_config.get("total_window_size", 48),
            "label_width": mltune_config.get("label_width", 1),
            "keras_tuner": mltune_config.get("keras_tuner", "Hyperband"),
            "hyperband_iterations": mltune_config.get("hyperband_iterations", 1),
            "max_epochs": mltune_config.get("max_epochs", 100),
            "min_epochs": mltune_config.get("min_epochs", 10),
            "epochs": mltune_config.get("epochs", 2),
            "num_trials": mltune_config.get("num_trials", 3),
            "overwrite": mltune_config.get("overwrite", True),
            "optimizer": mltune_config.get("optimizer", "adam"),
            "loss": mltune_config.get("loss", "mean_squared_error"),
            "metrics": mltune_config.get("metrics", ["mean_squared_error"]),
            "dropout": mltune_config.get("dropout", 0.2),
            "units": mltune_config.get("units", 32),
            "activation": mltune_config.get("activation", "relu"),
        }

        try:
            self.env.override_params({"mltune": mltune_overrides})
            logger.info("ML tuning parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override ML tuning parameters: %s", e)

    def _set_app_overrides(self):
        """Overrides APP tuning parameters if provided in config.yaml."""
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping APP tuning overrides.")
            return

        app_config = self.config.get("ML_TUNING_PARAMS", {})
        app_overrides = {
            "mp_app_primary_symbol": app_config.get("mp_app_primary_symbol", "EURUSD"),
            "mp_app_secondary_symbol": app_config.get("mp_app_secondary_symbol", "EURCHF"),
            "mp_app_broker": app_config.get("mp_app_broker", "METAQUOTES"),
            "mp_app_server": app_config.get("mp_app_server", ""),
            "mp_app_timeout": app_config.get("mp_app_timeout", 60000),
            "mp_app_portable": app_config.get("mp_app_portable", True),
            "mp_app_env": app_config.get("mp_app_env", "demo"),
            "mp_app_dfname1": app_config.get("mp_app_dfname1", "df_rates1"),
            "mp_app_dfname2": app_config.get("mp_app_dfname2", "df_rates2"),
            "mp_app_cfg_usedata": app_config.get("mp_app_cfg_usedata", 'df_file_rates'),
            "mp_app_rows": app_config.get("mp_app_rows", 2000),
            "mp_app_rowcount": app_config.get("mp_app_rowcount", 10000),
            "mp_app_ONNX_save": app_config.get("mp_app_ONNX_save", False),
            "mp_app_ml_show_plot": app_config.get("mp_app_ml_show_plot", False),
            "mp_app_ml_hard_run": app_config.get("mp_app_ml_hard_run", True),
            "mp_app_ml_tunemode": app_config.get("mp_app_ml_tunemode", True),
            "mp_app_ml_tunemodeepochs": app_config.get("mp_app_ml_tunemodeepochs", True),
            "mp_app_ml_Keras_tuner": app_config.get("mp_app_ml_Keras_tuner", 'hyperband'),
            "mp_app_ml_batch_size": app_config.get("mp_app_ml_batch_size", 4),
            "mp_app_ml_all_modelscale": app_config.get("mp_app_ml_all_modelscale", 2),
            "mp_app_ml_cnn_modelscale": app_config.get("mp_app_ml_cnn_modelscale", 2),
            "mp_app_ml_lstm_modelscale": app_config.get("mp_app_ml_lstm_modelscale", 2),
            "mp_app_ml_gru_modelscale": app_config.get("mp_app_ml_gru_modelscale", 2),
        }

        try:
            self.env.override_params({"app": app_overrides})
            logger.info("APP parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override APP parameters: %s", e)


# --------------------- Main usage example --------------------- #
if __name__ == "__main__":
    # Set up logging for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Create an instance of the CMqlOverrides class using the config file
    mql_overrides = CMqlOverrides("config.yaml")
    
    # Retrieve overridden parameters for inspection using the "params" attribute
    data_params = mql_overrides.params.get("data", {})
    feat_params = mql_overrides.params.get("feat", {})
    ml_params = mql_overrides.params.get("ml", {})
    mltune_params = mql_overrides.params.get("mltune", {})
    app_params = mql_overrides.params.get("app", {})
    
    print("Data Parameters:")
    for key, value in data_params.items():
        print(f"  {key}: {value}")
    
    print("\nFeature Parameters:")
    for key, value in feat_params.items():
        print(f"  {key}: {value}")
    
    print("\nML Parameters:")
    for key, value in ml_params.items():
        print(f"  {key}: {value}")
    
    print("\nML Tuning Parameters:")
    for key, value in mltune_params.items():
        print(f"  {key}: {value}")

    print("\nApp Parameters:")
    for key, value in app_params.items():
        print(f"  {key}: {value}")
