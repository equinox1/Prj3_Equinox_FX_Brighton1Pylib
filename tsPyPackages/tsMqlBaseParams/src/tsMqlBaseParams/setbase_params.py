#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: setbase_params.py
File: tsPyPackages/tsMqlBaseParams/src/tsMqlBaseParams/setbase_params.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
"""
import logging
logger = logging.getLogger(__name__)

import sys
import os
import platform
import json
from pathlib import Path
import yaml  # For loading configurations

# Initialize logger
logger = logging.getLogger(__name__)

# It’s assumed that these modules provide platform detection, logging, and other utilities.
from tsMqlPlatform import run_platform, platform_checker, config as global_config
from tsMqlEnvCore import CEnvCore

class CMqlEnvBaseParams(CEnvCore):
    """
    Environment setup for an MQL-based platform.
    
    This class loads configuration from a YAML or JSON file (if provided) and merges it
    with any keyword argument overrides. It then sets up paths for data, machine
    learning models, and MQL files based on the merged configuration.
    """
    def __init__(self, config_file: str = None, **kwargs):
        """
        Initialize environment parameters.

        Args:
            config_file (str, optional): Path to a YAML or JSON configuration file.
            **kwargs: Keyword arguments to override configuration settings.
        """
        # Load configuration from file if provided; otherwise use global_config if available.
        if config_file:
            try:
                # Determine file extension
                _, ext = os.path.splitext(config_file)
                if ext.lower() == '.json':
                    with open(config_file, 'r') as f:
                        file_config = json.load(f)
                else:
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
                file_config = {}
        else:
            # Use global_config if provided and is a dict, otherwise an empty dict.
            file_config = global_config if isinstance(global_config, dict) else {}

        # Merge file configuration with keyword arguments (kwargs take precedence)
        self.config = {**file_config, **kwargs}
        
        # Call parent initializer with custom parameters.
        super().__init__(custom_params=self.config)

        # Default platform directories based on OS
        self.DEFAULT_PLATFORM = {
            "Windows": {"basedir": "EQUINRUN"},
            "Linux": {"basedir": "EQUINRUNLIN"},
            "Darwin": {"basedir": "EQUINRUNMAC"},
        }
        
        # Initialize platform and state detection.
        self.pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = self.pchk.check_mql_state()
        
        if self.loadmql is None:
            logger.error("Failed to retrieve MQL state - check platform detection!")
        elif self.loadmql:
            logger.info("MQL state detected.")

        # Retrieve platform settings.
        self.mp_glob_pl_platform_type = self.os_platform
        plat_config = self.DEFAULT_PLATFORM.get(self.mp_glob_pl_platform_type, {})
        logger.info(f"Platform config: {plat_config}")
     
        self.mp_glob_pl_default_platform = plat_config.get("basedir")
        logger.info(f"Platform type: {self.mp_glob_pl_platform_type}")
        logger.info(f"Default platform: {self.mp_glob_pl_default_platform}")
        self.platform_dir = self.mp_glob_pl_default_platform

        # Retrieve broker placeholder but likely default to Metaquotes and override in main
        self.broker_name = self.config.get('mp_glob_sub_mql_broker_name', 'Metaquotes')

        # Setup base path.
        self.mp_glob_base_path = self._get_base_path()

        # Setup configuration paths: returns a single value.
        self.mp_glob_config_path = self._get_config_paths()

        # Setup log paths: returns a single value.
        self.mp_glob_log_path = self._get_log_paths()

        # Setup data paths: returns a single value.
        self.mp_glob_data_path = self._get_data_paths()

        # Setup ML paths: returns three values: base, models, and checkpoints.
        (self.model_base, self.project_dir, self.checkpoint_filepath) = self._get_ml_paths()

        # Setup MQL paths: returns base and data paths.
        (self.mql_basepath, self.mql_data_path, self.mql_include_path, self.mql_lib_path, self.mql_script_path, 
         self.mql_expert_path, self.mql_indicator_path) = self._get_mql_paths()

        # Ensure required attributes are initialized before use
        self.mp_glob_sub_ml_src_lib = self.config.get('mp_glob_sub_ml_src_lib', 'PythonLib')
        self.mp_glob_sub_ml_src_modeldata = self.config.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')
        self.model_uniq = self.config.get('mp_glob_sub_ml_baseuniq', '1')
        self.model_name = self.config.get('mp_glob_sub_ml_model_name', 'prjEquinox1_model')

        self.DEFAULT_PARAMS = {
            # Platform Path settings
            'mp_glob_base_pl_platform_type': self.mp_glob_pl_platform_type,
            'mp_glob_base_platform_dir': self.platform_dir,
            'mp_glob_base_pl_default_platform': self.mp_glob_pl_default_platform,
            'mp_glob_base_connect_path': self.mql_base_connect_path,   
            'mp_glob_base_config_path': self.mp_glob_config_path,
            'mp_glob_base_data_path': self.mp_glob_data_path,

            # log settings
            'mp_glob_base_log_path': self.mp_glob_log_path,
           
            # ML Base paths
            'mp_glob_base_path': self.mp_glob_base_path,
            'mp_glob_sub_ml_src_lib_lib_path': self.mp_glob_sub_ml_src_lib,
            'mp_glob_sub_ml_src_modeldata': self.mp_glob_sub_ml_src_modeldata,
            'mp_glob_base_ml_project_dir':  self.project_dir,
            'mp_glob_sub_ml_baseuniq': self.model_uniq,
            'mp_glob_sub_ml_model_name': self.model_name,

            # Checkpoint path
            'mp_glob_base_ml_checkpoint_filepath': self.checkpoint_filepath,
            # MQL paths
            'mp_glob_base_mql_basepath': self.mql_basepath,
            'mp_glob_base_mql_data_path': self.mql_data_path,
            'mp_glob_base_mql_include_path': self.mql_include_path,
            'mp_glob_base_mql_lib_path': self.mql_lib_path,
            'mp_glob_base_mql_script_path': self.mql_script_path,
            'mp_glob_base_mql_expert_path': self.mql_expert_path,
            'mp_glob_base_mql_indicator_path': self.mql_indicator_path,
            }

        # Log the distinct parameters
        logger.info("Distinct Base Environment parameters:")
        logger.info(f"mp_glob_base_path: {self.mp_glob_base_path}")
        logger.info(f"mp_glob_sub_ml_src_lib: {self.mp_glob_sub_ml_src_lib}")
        logger.info(f"mp_glob_sub_ml_src_modeldata: {self.mp_glob_sub_ml_src_modeldata}")
        logger.info(f"mp_glob_base_ml_project_dir: {self.project_dir}")
        logger.info(f"mp_glob_sub_ml_baseuniq: {self.model_uniq}")
        logger.info(f"mp_glob_sub_ml_model_name: {self.model_name}")

        logger.info("Distinct Base Environment parameters:")
        for key, value in self.DEFAULT_PARAMS.items():
            logger.info(f"{key}: {value}")

        self.params = self.DEFAULT_PARAMS
        logger.info(f"Base Environment parameters: {self.params}")

    def _get_base_path(self) -> Path:
        """
        Retrieve the main data path based on configuration settings.
        Uses the keys:
              mp_glob_sub_netdrive (default 'OneDrive')
            - mp_glob_sub_dir1 (default '8.0 Projects')
            - mp_glob_sub_dir2 (default '8.3 ProjectModelsEquinox')
            - mp_glob_sub_data (default 'Mql5Data')
              mp_glob_sub_ml_src_modeldata (default 'tsModelData')
              mp_glob_sub_ml_src_configdata (default 'tsConfigData')
              mp_glob_sub_ml_src_lib (default 'PythonLib')
              mp_glob_sub_ml_model_name (default 'prjEquinox1_prod.keras')
              mp_glob_sub_ml_baseuniq (default '1')
            - mp_glob_sub_mql_dir1 (default 'Mql5')
            - mp_glob_sub_mql_dir2 (default 'Files')
        """

        # Get the home directory and platform directory.
        one_drive = self.config.get('mp_glob_sub_netdrive', 'OneDrive')
        home_dir = Path.home()
        self.platformdir = self.mp_glob_pl_default_platform

        # Point the base directory to the correct location
        base_dir1 = self.config.get('mp_glob_sub_dir1', '8.0 Projects')
        if not base_dir1:
            logger.warning("Missing 'mp_glob_sub_dir1' in config. Using default: '8.0 Projects'.")

        base_dir2 = self.config.get('mp_glob_sub_dir2', '8.3 ProjectModelsEquinox')
        if not base_dir2:
            logger.warning("Missing 'mp_glob_sub_dir2' in config. Using default: '8.3 ProjectModelsEquinox'.")

        self.base_dir3 = home_dir / one_drive / base_dir1 / base_dir2 / self.platform_dir
        logger.info(f"Base path: {self.base_dir3}")

        # Construct the base path
        base_path = self.base_dir3
        logger.info(f"Base path: {base_path}")
        
        # Ensure the directory exists.
        base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base path resolved: {base_path}")

        return base_path

    def _get_config_paths(self):
         """
         Set up paths for configuration files.
         Creates directories for:
               - Base configuration directory (using platform directory)
               - Model data subdirectory
               - Configuration data subdirectory
               - Python library subdirectory
         """
         base_path = self.mp_glob_base_path
         self.config_subdir = self.config.get('mp_glob_sub_ml_src_configdata', 'tsConfigData')
         
         # Define the base configuration path.
         self.config_path = base_path / self.config_subdir
         Logflag = "Base:"
         logger.info(f"{Logflag} Config Path: {self.config_path}")

         logger.info(f"Configuration path: {self.config_path}")
        
         # Ensure directories exist.
         self.config_path.mkdir(parents=True, exist_ok=True)
        
         return self.config_path

    def _get_log_paths(self):
         """
         Set up paths for logging files.
         Creates directories for:
            - Log subdirectory
         """
         base_path = self.base_dir3
         # Define subdirectory name
         self.mp_glob_sub_log = self.config.get('mp_glob_sub_log', 'Logdir')
         # Define the log path.
         self.logdir = base_path / self.mp_glob_sub_log 
         Logflag = "Base:"
         logger.info(f"{Logflag} Log Path: {self.logdir}")

         # Ensure directories exist.
         self.logdir.mkdir(parents=True, exist_ok=True)
        
         return self.logdir

    def _get_data_paths(self):
         """
         Set up paths for machine learning files.
         Creates directories for:
            - Data subdirectory
         """
         base_path = self.base_dir3
         # Define subdirectory name
         self.mp_glob_sub_data = self.config.get('mp_glob_sub_data', 'Mql5Data')
         # Define the data path.
         self.data_path = base_path / self.mp_glob_sub_data 
         Logflag = "Base:"
         logger.info(f"{Logflag} Data Path: {self.data_path}")

         # Ensure directories exist.
         self.data_path.mkdir(parents=True, exist_ok=True)
        
         return self.data_path

    def _get_ml_paths(self):
         """
         Set up paths for machine learning files.
         Creates directories for:
            - Base ML directory (using platform directory)
            - Models subdirectory
            - Checkpoints subdirectory
         """
         base_path = self.base_dir3
         # Define subdirectory names
         self.lib_subdir = self.config.get('mp_glob_sub_ml_src_lib', 'PythonLib')
         self.mp_glob_sub_ml_src_lib = base_path / self.lib_subdir
         self.model_subdir = self.config.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')
         self.project_dir = self.config.get('mp_glob_sub_ml_directory', 'tshybrid_ensemble_tuning_prod')
         self.model_uniq = self.config.get('mp_glob_sub_ml_baseuniq', '1')
         self.model_name = self.config.get('mp_glob_sub_ml_model_name', 'prjEquinox1_model')

         # Define paths
         self.pybase = self.lib_subdir
         
         self.model_base = base_path / self.pybase / self.model_subdir
         self.project_dir = base_path / self.pybase / self.model_subdir / self.project_dir / self.model_uniq
         # project_name is intended as a string name, not a directory
         self.project_name = self.model_name
         self.checkpoint_filepath = self.project_dir 
         
         Logflag = "Base:"
         logger.info(f"{Logflag} Model base: {self.model_base}")
         logger.info(f"{Logflag} Project directory: {self.project_dir}")
         logger.info(f"{Logflag} Project name: {self.project_name}")
         logger.info(f"{Logflag} Checkpoint path: {self.checkpoint_filepath}")
         
         # Ensure directories exist.
         self.model_base.mkdir(parents=True, exist_ok=True)
         self.project_dir.mkdir(parents=True, exist_ok=True)
         # Removed mkdir call for self.project_name since it's a string.
         self.checkpoint_filepath.mkdir(parents=True, exist_ok=True)

         return self.model_base, self.project_dir, self.checkpoint_filepath

    def _get_mql_paths(self):
         base_path = self.base_dir3

         self.mql_base_ver = self.config.get('mp_glob_mql5_base_ver', 'MQL5')
         self.mql_base_broker = self.config.get('mp_glob_mql5_base_broker', 'Brokers')
         self.mql_base_broker_name = self.broker_name
         self.mql_dir1 = self.config.get('mp_glob_sub_mql_dir1', 'Mql5')
         self.mql_dir2 = self.config.get('mp_glob_sub_mql_dir2', 'Files')
         self.mql_dir3 = self.config.get('mp_glob_sub_mql_dir3', 'Include')
         self.mql_dir4 = self.config.get('mp_glob_sub_mql_dir4', 'Libraries')
         self.mql_dir5 = self.config.get('mp_glob_sub_mql_dir5', 'Scripts')
         self.mql_dir6 = self.config.get('mp_glob_sub_mql_dir6', 'Experts')
         self.mql_dir7 = self.config.get('mp_glob_sub_mql_dir7', 'Indicators')

         self.mql_basepath = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1
         self.mql_base_connect_path = base_path / self.mql_base_ver
         self.mql_data_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir2
         self.mql_include_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir3
         self.mql_lib_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir4
         self.mql_script_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir5
         self.mql_expert_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir6
         self.mql_indicator_path = base_path / self.mql_base_ver / self.mql_base_broker / self.mql_base_broker_name / self.mql_dir1 / self.mql_dir7

         Logflag = "Base:"
         logger.info(f"{Logflag} MQL base path: {self.mql_basepath}")
         logger.info(f"{Logflag} MQL connect path: {self.mql_base_connect_path}")
         logger.info(f"{Logflag} MQL data path: {self.mql_data_path}")
         logger.info(f"{Logflag} MQL include path: {self.mql_include_path}")
         logger.info(f"{Logflag} MQL lib path: {self.mql_lib_path}")
         logger.info(f"{Logflag} MQL script path: {self.mql_script_path}")
         logger.info(f"{Logflag} MQL expert path: {self.mql_expert_path}")
         logger.info(f"{Logflag} MQL indicator path: {self.mql_indicator_path}")

         # Ensure directories exist.
         self.mql_basepath.mkdir(parents=True, exist_ok=True)
         self.mql_data_path.mkdir(parents=True, exist_ok=True)
         self.mql_include_path.mkdir(parents=True, exist_ok=True)
         self.mql_lib_path.mkdir(parents=True, exist_ok=True)
         self.mql_script_path.mkdir(parents=True, exist_ok=True)
         self.mql_expert_path.mkdir(parents=True, exist_ok=True)
         self.mql_indicator_path.mkdir(parents=True, exist_ok=True)

         return (self.mql_basepath, self.mql_data_path, self.mql_include_path,
                 self.mql_lib_path, self.mql_script_path, self.mql_expert_path,
                 self.mql_indicator_path)

# Example usage:
if __name__ == "__main__":
    # You can provide a config file path and/or override parameters directly.
    env = CMqlEnvBaseParams(config_file='config.yaml', mp_glob_sub_dir1='My Projects')
    print(env.params)
