"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: setbase_params.py
File: tsPyPackages/tsMqlBaseParams/src/tsMqlBaseParams/setbase_params.py
Description: Set base parameters
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

import sys
import platform
from pathlib import Path
import os
import yaml  # For loading configurations
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config

from tsMqlDataParams import CMqlEnvDataParams
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLTunerParams import CMqlEnvMLTunerParams

# Initialize platform check
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")



class CMqlEnvBaseParams(BaseParamManager):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mp_glob_pl_platform_base=get_platform_helper()
        self.mp_glob_data_path=data_path_helper()
        self,modeldatapath=self.mp_glob_data_path
        
        self.mp_glob_ml_def_base_path = self.ml_path_helper()
        self.mp_glob_ml_num_base_path = self.ml_path_helper()
        self.mp_glob_ml_checkpoint_filepath = self.ml_path_helper()

        # MQL base path
        self.mp_glob_mql_basepath =self.mql_path_helper
        self.mp_glob_mql_data_path = self.mp_glob_data_path

        DEFAULT_PARAMS = {
            'mp_glob_pl_platform_base': self.mp_glob_pl_platform_base,
            'mp_glob_data_path': self.mp_glob_data_path,
            'mp_glob_ml_def_base_path': self.mp_glob_ml_def_base_path,
            'mp_glob_ml_num_base_path': self.mp_glob_ml_num_base_path,
            'mp_glob_ml_checkpoint_filepath': self.mp_glob_ml_checkpoint_filepath,
            'mp_glob_mql_basepath': self.mp_glob_mql_basepath,
            'mp_glob_mql_data_path': self.mp_glob_mql_data_path   
         }


    def  get_platform_helper():
      # platform settings and mql checks
        self.config = config  # Store config for later use
        self.config_platform = config.get("default_platform", None)  # equinrun for Windows, equinrunlin for Linux, equinrunmac for MacOS
        logger.info(f"Setting up global environment for platform: {self.config_platform}")

        # Set platform base
        self.mp_glob_pl_platform_base = self.config_platform
        logger.info(f"Platform base: {self.mp_glob_pl_platform_base}")
        return self.mp_glob_pl_platform_base


    def data_path_helper():
        # Use pathlib for cross-platform path handling
        self.mp_glob_home_dir = Path.home()

        # General environment paths (Using provided kwargs or config defaults)
        self.mp_glob_home_dir = kwargs.get('mp_glob_home_dir', self.mp_glob_home_dir)
        self.mp_glob_dir1 = kwargs.get('mp_glob_dir1', config.get('mp_glob_dir1', '8.0 Projects'))
        self.mp_glob_dir2 = kwargs.get('mp_glob_dir2', config.get('mp_glob_dir2', '8.3 ProjectModelsEquinox'))
        self.mp_glob_platform_dir = kwargs.get('mp_glob_platform_dir', config.get('mp_glob_platform_dir', self.mp_glob_pl_platform_base))
        self.mp_glob_data_dir = kwargs.get('mp_glob_data_dir', config.get('data_dir', 'Data'))

        # Construct the data path
        self.mp_glob_data_path = Path(self.mp_glob_home_dir / self.mp_glob_dir1 / self.mp_glob_dir2 / self.mp_glob_platform_dir / self.mp_glob_data_dir)
        logger.info(f"Data path: {self.mp_glob_data_path}")

        self.modeldatapath = self.mp_glob_data_path  # Alias for compatibility
   

    def  ml_path_helper():
        # ML base path
        self.mp_glob_ml_src_base = Path(self.mp_glob_home_dir/ self.mp_glob_dir1 / self.mp_glob_dir2)
        self.mp_glob_ml_def_base_path = self.mp_glob_ml_src_base / self.mp_glob_pl_platform_base / self.mp_glob_ml_src_lib / self.mp_glob_ml_src_modeldata / self.mp_glob_ml_directory
        self.mp_glob_ml_num_base_path = self.mp_glob_ml_def_base_path / self.mp_glob_ml_baseuniq
        self.mp_glob_ml_checkpoint_filepath = self.mp_glob_ml_def_base_path

        logger.info(f"TuneParams: Default base path: {self.mp_glob_ml_def_base_path}")
        logger.info(f"TuneParams: Numbered base path: {self.mp_glob_ml_num_base_path}")
        logger.info(f"TuneParams: Checkpoint file path: {self.mp_glob_ml_checkpoint_filepath}")

        # MQL base path
        self.mp_glob_mql_dir1 = config.get('mp_glob_mql_dir1', 'Mql5')
        self.mp_glob_mql_basepath = Path(self.mp_glob_home_dir/ self.mp_glob_dir1 / self.mp_glob_dir2 / self.mp_glob_platform_dir / self.mp_glob_mql_dir1)
        self.mp_glob_mql_data_path = self.mp_glob_data_path

        logger.info(f"MQL base path: {self.mp_glob_mql_basepath}")
        logger.info(f"MQL data path: {self.mp_glob_mql_data_path}")  
        return self.mp_glob_ml_def_base_path, self.mp_glob_ml_num_base_path, self.mp_glob_ml_checkpoint_filepath

    def mql_path_helper():
        self.mp_glob_mql_dir1 = config.get('mp_glob_mql_dir1', 'Mql5')
        self.mp_glob_mql_basepath = Path(self.mp_glob_home_dir/ self.mp_glob_dir1 / self.mp_glob_dir2 / self.mp_glob_platform_dir / self.mp_glob_mql_dir1)
        self.mp_glob_mql_data_path = self.mp_glob_data_path

        logger.info(f"MQL base path: {self.mp_glob_mql_basepath}")
        logger.info(f"MQL data path: {self.mp_glob_mql_data_path}")  
        return self.mp_glob_mql_basepath, self.mp_glob_mql_data_path 