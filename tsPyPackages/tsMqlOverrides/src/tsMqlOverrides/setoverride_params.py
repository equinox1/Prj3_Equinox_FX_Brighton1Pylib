#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: setoverride_params.py
File: tsPyPackages/tsMqlOverrides/tsMqlOverrides.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.2
"""

import os
from pathlib import Path
import yaml
import logging
import copy

# Basic logger configuration for demonstration
logger = logging.getLogger(__name__)

# Adjust these imports as necessary for your project structure
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlEnvCore import CEnvCore

def deep_merge(source, overrides):
    """
    Recursively merge two dictionaries. Keys in overrides are merged into source.
    """
    merged = copy.deepcopy(source)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

class CMqlOverrides(CEnvCore):
    """
    Environment setup for the MQL-based platform, handling paths and configurations.
    Reads configuration parameters from config.yaml and deep merges them with defaults.
    """

    def __init__(self, config_path="config.yaml", **kwargs):
        # Load configuration from YAML file
        self.config = self._load_config(config_path)
        # Pass additional kwargs to base class if needed
        super().__init__(custom_params=kwargs)
   
        # Initialize the environment manager and load default parameters.
        self.env = CMqlEnvMgr()
        # Get the default parameters from the environment manager.
        self.params = self.env.all_params()

        # Merge configuration overrides into the default parameters using deep merge.
        self.params = deep_merge(self.params, self.config)

        # Apply individual category overrides (if needed)
        self._set_data_overrides()
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
            logger.error("Overrides: Configuration file %s not found.", config_path)
            return {}

    def _set_data_overrides(self):
        """Overrides default data parameters using values from config.yaml if available."""
        data_config = self.config.get("DATA_PARAMS", {})
        data_overrides = {
            "mp_data_load": data_config.get("mp_data_load", True),
            "mp_data_save": data_config.get("mp_data_save", False),
            "mp_data_savefile": data_config.get("mp_data_savefile", False),
            # Additional data parameters…
        }
        try:
            # Perform a deep merge on the "data" category
            current = self.env.all_params().get("data", {})
            merged = deep_merge(current, data_overrides)
            self.env.override_params({"data": merged})
            logger.info("Data parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override data parameters: %s", e)

    def _set_ml_overrides(self):
        """Overrides ML-related parameters using config.yaml settings if available."""
        ml_config = self.config.get("ML_PARAMS", {})
        ml_overrides = {
            "mp_ml_tf_shiftin": ml_config.get("mp_ml_tf_shiftin", 1),
            "mp_ml_hl_avg_col": ml_config.get("mp_ml_hl_avg_col", "HLAvg"),
            # Additional ML parameters…
        }
        try:
            current = self.env.all_params().get("ml", {})
            merged = deep_merge(current, ml_overrides)
            self.env.override_params({"ml": merged})
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
            current = self.env.all_params().get("mltune", {})
            merged = deep_merge(current, mltune_overrides)
            self.env.override_params({"mltune": merged})
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
                current = self.env.all_params().get("mltune", {})
                merged = deep_merge(current, tuner_overrides)
                self.env.override_params({"mltune": merged})
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
            # Additional app parameters…
        }
        try:
            current = self.env.all_params().get("app", {})
            merged = deep_merge(current, app_overrides)
            self.env.override_params({"app": merged})
            logger.info("APP parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override APP parameters: %s", e)

# --------------------- Main usage example --------------------- #
if __name__ == "__main__":
    # Create an instance of the CMqlOverrides class.
    mql_overrides = CMqlOverrides()  # If no config.yaml is provided, only defaults are used
    
    # Retrieve parameters for inspection using the "params" attribute.
    data_params = mql_overrides.env.all_params().get("data", {})
    ml_params = mql_overrides.env.all_params().get("ml", {})
    mltune_params = mql_overrides.env.all_params().get("mltune", {})
    app_params = mql_overrides.env.all_params().get("app", {})
    
    print("Data Parameters:")
    for key, value in data_params.items():
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
