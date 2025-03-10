#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: etbase_params.py
File: tsPyPackages/tsMqlBaseParams/src/tsMqlBaseParams/setbase_params.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
"""
import sys
import platform
from pathlib import Path
import yaml  # For loading configurations
import logging

# It’s assumed that these modules provide platform detection, logging, and other utilities.
from tsMqlPlatform import run_platform, platform_checker, logger, config as global_config
from tsMqlEnvCore import CEnvCore

class CMqlEnvBaseParams(CEnvCore):
    """
    Environment setup for an MQL-based platform.
    
    This class loads configuration from a YAML file (if provided) and merges it
    with any keyword argument overrides. It then sets up paths for data, machine
    learning models, and MQL files based on the merged configuration.
    """
    def __init__(self, config_file: str = None, **kwargs):
        """
        Initialize environment parameters.

        Args:
            config_file (str, optional): Path to a YAML configuration file.
            **kwargs: Keyword arguments to override configuration settings.
        """
        # Load YAML configuration if provided; otherwise use global_config or an empty dict.
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Error loading YAML config from {config_file}: {e}")
                file_config = {}
        else:
            file_config = global_config if global_config else {}
        
        # Ensure file_config is a dictionary.
        if not isinstance(file_config, dict):
            try:
                # Attempt to convert to a dict (if global_config supports this)
                file_config = dict(file_config)
            except Exception as e:
                logger.error("global_config cannot be converted to a dict: %s", e)
                file_config = {}
        
        # Merge YAML config with kwargs (kwargs take precedence)
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

        # Setup paths.
        self.mp_glob_data_path = self._get_data_path()
        (self.mp_glob_ml_def_base_path,
         self.mp_glob_ml_num_base_path,
         self.mp_glob_ml_checkpoint_filepath) = self._get_ml_paths()
        (self.mp_glob_mql_basepath,
         self.mp_glob_mql_data_path) = self._get_mql_paths()

        # Additional model settings (if needed, they can also be overridden)
        self.mp_glob_model_data_path = self.config.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')
        self.mp_glob_model_config_path = self.config.get('mp_glob_sub_ml_src_configdata', 'tsConfigData')
        self.mp_glob_model_num_path = self.config.get('mp_glob_sub_ml_src_lib', 'PythonLib')
        self.mp_glob_model_checkpoint_path = self.config.get('mp_glob_sub_ml_model_name', 'prjEquinox1_prod.keras')

        self.DEFAULT_PARAMS = {
            'mp_glob_pl_platform_type': self.mp_glob_pl_platform_type,
            'mp_glob_pl_default_platform': self.mp_glob_pl_default_platform,
            'mp_glob_data_path': str(self.mp_glob_data_path),
            'mp_glob_ml_def_base_path': str(self.mp_glob_ml_def_base_path),
            'mp_glob_ml_num_base_path': str(self.mp_glob_ml_num_base_path),
            'mp_glob_ml_checkpoint_filepath': str(self.mp_glob_ml_checkpoint_filepath),
            'mp_glob_mql_basepath': str(self.mp_glob_mql_basepath),
            'mp_glob_mql_data_path': str(self.mp_glob_mql_data_path),
            'mp_glob_model_data_path': self.mp_glob_model_data_path,
            'mp_glob_model_config_path': self.mp_glob_model_config_path,
            'mp_glob_model_num_path': self.mp_glob_model_num_path,
            'mp_glob_model_checkpoint_path': self.mp_glob_model_checkpoint_path,
            }
        logger.info("Distinct Base Environment parameters:")

        logger.info(f"mp_glob_pl_platform_type: {self.mp_glob_pl_platform_type}")
        logger.info(f"mp_glob_pl_default_platform: {self.mp_glob_pl_default_platform}")
        logger.info(f"mp_glob_data_path: {str(self.mp_glob_data_path)}")
        logger.info(f"mp_glob_ml_def_base_path: {str(self.mp_glob_ml_def_base_path)}")
        logger.info(f"mp_glob_ml_num_base_path: {str(self.mp_glob_ml_num_base_path)}")
        logger.info(f"mp_glob_ml_checkpoint_filepath: {str(self.mp_glob_ml_checkpoint_filepath)}")
        logger.info(f"mp_glob_mql_basepath: {str(self.mp_glob_mql_basepath)}")
        logger.info(f"mp_glob_mql_data_path: {str(self.mp_glob_mql_data_path)}")
        logger.info(f"mp_glob_model_data_path: {self.mp_glob_model_data_path}")
        logger.info(f"mp_glob_model_config_path: {self.mp_glob_model_config_path}")
        logger.info(f"mp_glob_model_num_path: {self.mp_glob_model_num_path}")
        logger.info(f"mp_glob_model_checkpoint_path: {self.mp_glob_model_checkpoint_path}")


        self.params = self.DEFAULT_PARAMS
        logger.info(f"Base Environment parameters: {self.params}")

    def _get_data_path(self) -> Path:
        """
        Retrieve the main data path based on configuration settings.
        Uses the keys:
            - mp_glob_sub_dir1 (default '8.0 Projects')
            - mp_glob_sub_dir2 (default '8.3 ProjectModelsEquinox')
            - mp_glob_sub_data (default 'Mql5Data')
            - mp_glob_sub_netdrive (default 'OneDrive')
        """
        base_dir = self.config.get('mp_glob_sub_dir1', '8.0 Projects')
        if not base_dir:
            logger.warning("Missing 'mp_glob_sub_dir1' in config. Using default: '8.0 Projects'.")

        project_dir = self.config.get('mp_glob_sub_dir2', '8.3 ProjectModelsEquinox')
        if not project_dir:
            logger.warning("Missing 'mp_glob_sub_dir2' in config. Using default: '8.3 ProjectModelsEquinox'.")

        data_subdir = self.config.get('mp_glob_sub_data', 'Mql5Data')
        home_dir = Path.home()
        one_drive = self.config.get('mp_glob_sub_netdrive', 'OneDrive')
        if not one_drive:
            logger.warning("Missing 'mp_glob_sub_netdrive' in config. Using default: 'OneDrive'.")

        data_path = home_dir / one_drive / base_dir / project_dir / self.platform_dir / data_subdir
        logger.info(f"Data path: {data_path}")
        
        # Ensure the directory exists.
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data path resolved: {data_path}")

        return data_path
       
    def _get_ml_paths(self):
         """
         Set up paths for machine learning files.
         Creates directories for:
            - Base ML directory (using platform directory)
            - Models subdirectory
            - Checkpoints subdirectory
         """
         base_path = self.mp_glob_data_path.parent
         ml_def_path = base_path / self.platform_dir
         ml_num_path = ml_def_path / "models"
         
         # Define the checkpoints path explicitly.
         ml_checkpoint_path = ml_def_path / "checkpoints"
         
         # Log the platform type using the existing logger.
         logger.info('mp_glob_pl_platform_type')
         
         # Optionally, if you need to store the checkpoint path in the logger, you might do:
         logger._path = ml_checkpoint_path

         # Ensure directories exist.
         ml_def_path.mkdir(parents=True, exist_ok=True)
         ml_num_path.mkdir(parents=True, exist_ok=True)
         ml_checkpoint_path.mkdir(parents=True, exist_ok=True)

         return ml_def_path, ml_num_path, ml_checkpoint_path

       
    def _get_mql_paths(self):
        """
        Configure paths for MQL-specific directories using:
            - mp_glob_sub_mql_dir1 (default 'Mql5')
            - mp_glob_sub_mql_dir2 (default 'Files')
        Constructs the MQL base path from the parent of the data path.
        """
        mql_dir = self.config.get('mp_glob_sub_mql_dir1', 'Mql5')
        mql_subdir = self.config.get('mp_glob_sub_mql_dir2', 'Files')
        
        mql_basepath = self.mp_glob_data_path.parent / mql_dir / mql_subdir
        mql_data_path = self.mp_glob_data_path

        # Ensure directories exist.
        mql_basepath.mkdir(parents=True, exist_ok=True)
        mql_data_path.mkdir(parents=True, exist_ok=True)

        return mql_basepath, mql_data_path

# Example usage:
if __name__ == "__main__":
    # You can provide a config file path and/or override parameters directly.
    env = CMqlEnvBaseParams(config_file='config.yaml', mp_glob_sub_dir1='My Projects')
    print(env.params)
