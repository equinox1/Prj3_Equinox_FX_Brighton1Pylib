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
from pathlib import Path
import yaml

from tsMqlPlatform import run_platform, platform_checker, get_config
from tsMqlBaseParams.config import BaseConfig
from tsMqlEnvCore import CEnvCore

class CMqlEnvBaseParams(CEnvCore):
    def __init__(self, config_file: str = None, **kwargs):
        self.config = BaseConfig()
        logger.info("BaseParams: Loading configuration...")
        logger.info(f"BaseParams: Looking for user provided config file: {self.config.config_file}")

        self.DEFAULT_PLATFORM = {
            "Windows": {"basedir": "EQUINRUN"},
            "Linux": {"basedir": "EQUINRUNLIN"},
            "Darwin": {"basedir": "EQUINRUNMAC"},
        }

        self.pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = self.pchk.check_mql_state()

        if self.loadmql is None:
            logger.error("Failed to retrieve MQL state - check platform detection!")
        elif self.loadmql:
            logger.info("MQL state detected.")

        self.mp_glob_pl_platform_type = self.os_platform
        plat_config = self.DEFAULT_PLATFORM.get(self.mp_glob_pl_platform_type, {})
        logger.info(f"Platform config: {plat_config}")

        self.mp_glob_pl_default_platform = plat_config.get("basedir")
        logger.info(f"Platform type: {self.mp_glob_pl_platform_type}")
        logger.info(f"Default platform: {self.mp_glob_pl_default_platform}")

        self.broker_name = self.config.get('mp_glob_sub_mql_broker_name', 'Metaquotes')

        self.mp_glob_base_path = self._get_base_path()
        self.mp_glob_config_path = self._get_config_paths()
        self.mp_glob_log_path = self._get_log_paths()
        self.mp_glob_data_path = self._get_data_paths()
        (self.model_base, self.project_dir, self.checkpoint_filepath) = self._get_ml_paths()
        (self.mql_basepath, self.mql_data_path, self.mql_include_path, self.mql_lib_path,
         self.mql_script_path, self.mql_expert_path, self.mql_indicator_path) = self._get_mql_paths()

        self.mp_glob_sub_ml_src_lib = self.config.get('mp_glob_sub_ml_src_lib', 'PythonLib')
        self.mp_glob_sub_ml_src_modeldata = self.config.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')
        self.model_uniq = self.config.get('mp_glob_sub_ml_baseuniq', '3')
        self.model_name = self.config.get('mp_glob_sub_ml_model_name', 'prjEquinox1_model')

        self.DEFAULT_PARAMS = {
            'mp_glob_base_pl_platform_type': self.mp_glob_pl_platform_type,
            'mp_glob_base_platform_dir': self.platform_dir,
            'mp_glob_base_pl_default_platform': self.mp_glob_pl_default_platform,
            'mp_glob_base_connect_path': self.mql_base_connect_path,   
            'mp_glob_base_config_path': self.mp_glob_config_path,
            'mp_glob_base_data_path': self.mp_glob_data_path,
            'mp_glob_base_log_path': self.mp_glob_log_path,
            'mp_glob_base_path': self.mp_glob_base_path,
            'mp_glob_sub_ml_src_lib_lib_path': self.mp_glob_sub_ml_src_lib,
            'mp_glob_sub_ml_src_modeldata': self.model_base,
            'mp_glob_base_ml_project_dir': self.project_dir,
            'mp_glob_sub_ml_baseuniq': self.model_uniq,
            'mp_glob_sub_ml_model_name': self.model_name,
            'mp_glob_base_ml_checkpoint_filepath': self.checkpoint_filepath,
            'mp_glob_base_mql_basepath': self.mql_basepath,
            'mp_glob_base_mql_data_path': self.mql_data_path,
            'mp_glob_base_mql_include_path': self.mql_include_path,
            'mp_glob_base_mql_lib_path': self.mql_lib_path,
            'mp_glob_base_mql_script_path': self.mql_script_path,
            'mp_glob_base_mql_expert_path': self.mql_expert_path,
            'mp_glob_base_mql_indicator_path': self.mql_indicator_path,
        }

        logger.info("Distinct Base Environment parameters:")
        for key, value in self.DEFAULT_PARAMS.items():
            logger.info(f"{key}: {value}")

        self.params = self.DEFAULT_PARAMS
        logger.info(f"Base Environment parameters: {self.params}")

    def _get_base_path(self) -> Path:
        mp_glob_runtype = self.config.get('mp_glob_runtype', 'onedrive')
        if self.os_platform == "Windows" and mp_glob_runtype == "onedrive":
            drive = self.config.get('mp_glob_sub_win_drive', 'C:')
            user_mnt = self.config.get('mp_glob_sub_win_user_mnt', 'Users')
            user = self.config.get('mp_glob_sub_win_user', 'shepa')
            netdrive = self.config.get('mp_glob_sub_netdrive', 'OneDrive')
            self.platform_dir = os.path.join(drive, user_mnt, user, netdrive)

        elif self.os_platform == "Windows" and mp_glob_runtype == "localdrive":
            drive = self.config.get('mp_glob_sub_win_drive', 'C:')
            netdrive = self.config.get('mp_glob_sub_netdrive', 'WinRunmnt1')
            self.platform_dir = os.path.join(drive, netdrive)

        elif self.os_platform == "Linux":
            self.platform_dir = self.config.get('mp_glob_sub_netdrive', 'LinuxRunmnt1')

        elif self.os_platform == "Darwin":
            self.platform_dir = self.config.get('mp_glob_sub_netdrive', 'MacRunmnt1')

        else:
            logger.error(f"Unsupported platform: {self.os_platform}.")
            raise EnvironmentError("Unsupported platform.")

        base_dir1 = self.config.get('mp_glob_sub_dir1', '8.0 Projects')
        base_dir2 = self.config.get('mp_glob_sub_dir2', '8.3 ProjectModelsEquinox')

        self.base_dir3 = Path(self.platform_dir) / base_dir1 / base_dir2 / self.mp_glob_pl_default_platform
        logger.info(f"Base path: {self.base_dir3}")

        self.base_dir3.mkdir(parents=True, exist_ok=True)
        return self.base_dir3

    def _get_config_paths(self):
        base_path = self.mp_glob_base_path
        config_subdir = self.config.get('mp_glob_sub_ml_src_configdata', 'tsConfigData')
        config_path = base_path / config_subdir
        logger.info(f"Config Path: {config_path}")
        config_path.mkdir(parents=True, exist_ok=True)
        return config_path

    def _get_log_paths(self):
        base_path = self.base_dir3
        log_subdir = self.config.get('mp_glob_sub_log', 'Logdir')
        log_path = base_path / log_subdir
        logger.info(f"Log Path: {log_path}")
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    def _get_data_paths(self):
        base_path = self.base_dir3
        data_subdir = self.config.get('mp_glob_sub_data', 'Mql5Data')
        data_path = base_path / data_subdir
        logger.info(f"Data Path: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

    def _get_ml_paths(self):
        base_path = self.base_dir3
        lib_subdir = self.config.get('mp_glob_sub_ml_src_lib', 'PythonLib')
        model_subdir = self.config.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')
        project_subdir = self.config.get('mp_glob_sub_ml_directory', 'tshybrid_ensemble_tuning_prod')
        model_uniq = self.config.get('mp_glob_sub_ml_baseuniq', '1')
        model_name = self.config.get('mp_glob_sub_ml_model_name', 'prjEquinox1_model')

        model_base = base_path / lib_subdir / model_subdir
        project_dir = model_base / project_subdir / model_uniq
        checkpoint_filepath = project_dir

        model_base.mkdir(parents=True, exist_ok=True)
        project_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_filepath.mkdir(parents=True, exist_ok=True)

        return model_base, project_dir, checkpoint_filepath

    def _get_mql_paths(self):
        base_path = self.base_dir3

        mql_base_ver = self.config.get('mp_glob_mql5_base_ver', 'MQL5')
        mql_base_broker = self.config.get('mp_glob_mql5_base_broker', 'Brokers')
        mql_base_broker_name = self.broker_name
        mql_dirs = {
            "dir1": self.config.get('mp_glob_sub_mql_dir1', 'Mql5'),
            "dir2": self.config.get('mp_glob_sub_mql_dir2', 'Files'),
            "dir3": self.config.get('mp_glob_sub_mql_dir3', 'Include'),
            "dir4": self.config.get('mp_glob_sub_mql_dir4', 'Libraries'),
            "dir5": self.config.get('mp_glob_sub_mql_dir5', 'Scripts'),
            "dir6": self.config.get('mp_glob_sub_mql_dir6', 'Experts'),
            "dir7": self.config.get('mp_glob_sub_mql_dir7', 'Indicators'),
        }

        base = base_path / mql_base_ver / mql_base_broker / mql_base_broker_name / mql_dirs["dir1"]
        self.mql_base_connect_path = base_path / mql_base_ver

        paths = {
            "data": base / mql_dirs["dir2"],
            "include": base / mql_dirs["dir3"],
            "lib": base / mql_dirs["dir4"],
            "script": base / mql_dirs["dir5"],
            "expert": base / mql_dirs["dir6"],
            "indicator": base / mql_dirs["dir7"],
        }

        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"MQL {name} path: {path}")

        return (base, paths["data"], paths["include"], paths["lib"],
                paths["script"], paths["expert"], paths["indicator"])

# Example usage
if __name__ == "__main__":
    env = CMqlEnvBaseParams(config_file='config.yaml', mp_glob_sub_dir1='My Projects')
    print(env.params)
