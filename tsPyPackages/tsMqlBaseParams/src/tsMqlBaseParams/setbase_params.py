import sys
import platform
from pathlib import Path
import os
import yaml  # For loading configurations
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
from tsMqlParamMgr import BaseParamManager

# Initialize platform check
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")

class CMqlEnvBaseParams(BaseParamManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.config = config  # Store config for later use
        self.config_platform = config.get("default_platform", None)  # Platform configuration
        logger.info(f"Setting up global environment for platform: {self.config_platform}")

        self.mp_glob_pl_platform_base = self.get_platform_helper()
        self.mp_glob_data_path = self.data_path_helper()
        self.mp_glob_ml_def_base_path, self.mp_glob_ml_num_base_path, self.mp_glob_ml_checkpoint_filepath = self.ml_path_helper()
        self.mp_glob_mql_basepath, self.mp_glob_mql_data_path = self.mql_path_helper()

        self.DEFAULT_PARAMS = {
            'mp_glob_pl_platform_base': self.mp_glob_pl_platform_base,
            'mp_glob_data_path': self.mp_glob_data_path,
            'mp_glob_ml_def_base_path': self.mp_glob_ml_def_base_path,
            'mp_glob_ml_num_base_path': self.mp_glob_ml_num_base_path,
            'mp_glob_ml_checkpoint_filepath': self.mp_glob_ml_checkpoint_filepath,
            'mp_glob_mql_basepath': self.mp_glob_mql_basepath,
            'mp_glob_mql_data_path': self.mp_glob_mql_data_path
        }

    def get_platform_helper(self):
        logger.info(f"Platform base: {self.config_platform}")
        return self.config_platform

    def data_path_helper(self):
        self.mp_glob_home_dir = Path.home()
        
        self.mp_glob_home_dir = self.kwargs.get('mp_glob_home_dir', self.mp_glob_home_dir)
        self.mp_glob_dir1 = self.kwargs.get('mp_glob_dir1', config.get('mp_glob_dir1', '8.0 Projects'))
        self.mp_glob_dir2 = self.kwargs.get('mp_glob_dir2', config.get('mp_glob_dir2', '8.3 ProjectModelsEquinox'))
        self.mp_glob_platform_dir = self.kwargs.get('mp_glob_platform_dir', config.get('mp_glob_platform_dir', self.mp_glob_pl_platform_base))
        self.mp_glob_data_dir = self.kwargs.get('mp_glob_data_dir', config.get('data_dir', 'Data'))

        # Construct the data path
        data_path = Path(self.mp_glob_home_dir / self.mp_glob_dir1 / self.mp_glob_dir2 / self.mp_glob_platform_dir / self.mp_glob_data_dir)
        logger.info(f"Data path: {data_path}")
        return data_path

    def ml_path_helper(self):
        self.mp_glob_ml_src_base = Path(self.mp_glob_home_dir / self.mp_glob_dir1 / self.mp_glob_dir2)
        self.mp_glob_ml_def_base_path = self.mp_glob_ml_src_base / self.mp_glob_pl_platform_base
        self.mp_glob_ml_num_base_path = self.mp_glob_ml_def_base_path / "models"
        self.mp_glob_ml_checkpoint_filepath = self.mp_glob_ml_def_base_path / "checkpoints"

        logger.info(f"ML default base path: {self.mp_glob_ml_def_base_path}")
        logger.info(f"ML numbered base path: {self.mp_glob_ml_num_base_path}")
        logger.info(f"ML checkpoint file path: {self.mp_glob_ml_checkpoint_filepath}")

        return self.mp_glob_ml_def_base_path, self.mp_glob_ml_num_base_path, self.mp_glob_ml_checkpoint_filepath

    def mql_path_helper(self):
        self.mp_glob_mql_dir1 = config.get('mp_glob_mql_dir1', 'Mql5')
        mql_basepath = Path(self.mp_glob_home_dir / self.mp_glob_dir1 / self.mp_glob_dir2 / self.mp_glob_platform_dir / self.mp_glob_mql_dir1)
        mql_data_path = self.mp_glob_data_path

        logger.info(f"MQL base path: {mql_basepath}")
        logger.info(f"MQL data path: {mql_data_path}")
        
        return mql_basepath, mql_data_path

# Instantiate the class
base_global_setter = CMqlEnvBaseParams()
