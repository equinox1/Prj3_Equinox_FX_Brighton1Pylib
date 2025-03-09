#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: setoverride_params.py
File: tsPyPackages/tsMqlOverrides/tsMqlOverrides.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

import os
from pathlib import Path
import yaml
import logging

# For demonstration purposes, we use a basic logger here.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tsMqlOverrides")

# These are your environment manager and base class imports.
# Adjust the imports as necessary for your project structure.
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlEnvCore import CEnvCore

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
   
        # Initialize the environment manager and load default parameters.
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()

        # Call the override methods.
        self._initialize_env_mgr()
        self._set_data_overrides()
        self._set_feature_overrides()
        self._set_ml_overrides()
        self._set_mltune_overrides()
        self._set_tuner_overrides()
        self._set_app_overrides()

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
        """Initializes the environment manager and loads parameters."""
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
        data_config = self.config.get("data", {})
        data_overrides = {
            "mp_data_load": data_config.get("mp_data_load", True),
            "mp_data_save": data_config.get("mp_data_save", False),
            "mp_data_savefile": data_config.get("mp_data_savefile", False),
            # Add additional data overrides as needed...
        }
        try:
            self.env.override_params({"data": data_overrides})
            logger.info("Data parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override data parameters: %s", e)

    def _set_feature_overrides(self):
        """Overrides feature parameters using settings from config.yaml if provided."""
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
        ml_config = self.config.get("ML_PARAMS", {})
        ml_overrides = {
            "mp_ml_tf_shiftin": ml_config.get("mp_ml_tf_shiftin", 1),
            "mp_ml_hl_avg_col": ml_config.get("mp_ml_hl_avg_col", "HLAvg"),
            # Add additional ML overrides as needed...
        }
        try:
            self.env.override_params({"ml": ml_overrides})
            logger.info("ML parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override ML parameters: %s", e)

    def _set_mltune_overrides(self):
        """Overrides ML tuning parameters if provided in config.yaml."""
        mltune_config = self.config.get("ML_TUNING_PARAMS", {})
        mltune_overrides = {
            "mp_ml_tunemode": mltune_config.get("mp_ml_tunemode", True),
        }
        try:
            self.env.override_params({"mltune": mltune_overrides})
            logger.info("ML tuning parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override ML tuning parameters: %s", e)

    def _set_tuner_overrides(self):
        """Overrides tuner-specific parameters using settings from config.yaml if provided."""
        tuner_config = self.config.get("TUNER_PARAMS", {})
        if tuner_config:
            tuner_overrides = {
                "tunemode": tuner_config.get("tunemode", "Hyperband"),
                "tunemodeepochs": tuner_config.get("tunemodeepochs", 100),
                "seed": tuner_config.get("seed", 42),
                "keras_tuner": tuner_config.get("keras_tuner", "Hyperband"),
                "hyperband_iterations": tuner_config.get("hyperband_iterations", 1),
            }
            try:
                self.env.override_params({"mltune": tuner_overrides})
                logger.info("Tuner parameters overridden successfully.")
            except Exception as e:
                logger.error("Failed to override tuner parameters: %s", e)
        else:
            logger.info("No tuner overrides provided.")

    def _set_app_overrides(self):
        """Overrides APP tuning parameters if provided in config.yaml."""
        app_config = self.config.get("ML_TUNING_PARAMS", {})
        app_overrides = {
            "mp_app_primary_symbol": app_config.get("mp_app_primary_symbol", "EURUSD"),
            "mp_app_secondary_symbol": app_config.get("mp_app_secondary_symbol", "EURCHF"),
            # Add additional app overrides as needed...
        }
        try:
            self.env.override_params({"app": app_overrides})
            logger.info("APP parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override APP parameters: %s", e)

# --------------------- Main usage example --------------------- #
if __name__ == "__main__":
    # Create an instance of the CMqlOverrides class (using default config or one specified)
    mql_overrides = CMqlOverrides()  # If no config.yaml is found, defaults are used
    
    # Retrieve parameters for inspection using the "params" attribute
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
    
    print("\nML Tuning & Tuner Parameters:")
    for key, value in mltune_params.items():
        print(f"  {key}: {value}")
    
    print("\nApp Parameters:")
    for key, value in app_params.items():
        print(f"  {key}: {value}")
