import sys
import platform
import os
from pathlib import Path
import yaml  # For loading configurations
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
from tsMqlEnvCore import CEnvCore

class CMqlEnvBaseParams(CEnvCore):
    """
    Environment setup for MQL-based platform, handling paths and configurations.
    """
    def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization
        self.config = config  # Load configuration

        self.DEFAULT_PLATFORM = {
            "Windows": {"basedir": "EQUINRUN"},
            "Linux": {"basedir": "EQUINRUNLIN"},
            "Darwin": {"basedir": "EQUINRUNMAC"},
        }
        
        # Initialize platform and state
        self.pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = self.pchk.check_mql_state()
        
        if self.loadmql is None:
            logger.error("Failed to retrieve MQL state - check platform detection!")
        elif self.loadmql:
            logger.info("MQL state detected.")

        # Retrieve platform settings
       
        self.mp_glob_pl_platform_type = self.os_platform
        plat_config = self.DEFAULT_PLATFORM.get(self.mp_glob_pl_platform_type, {})
        logger.info(f"Platform config: {plat_config}")
     
        self.mp_glob_pl_default_platform = plat_config.get("basedir")

        logger.info(f"Platform type: {self.mp_glob_pl_platform_type}")
        logger.info(f"Default platform: {self.mp_glob_pl_default_platform}")
        self.platform_dir=self.mp_glob_pl_default_platform

        # Setup paths
        self.mp_glob_data_path = self._get_data_path()
        self.mp_glob_ml_def_base_path, self.mp_glob_ml_num_base_path, self.mp_glob_ml_checkpoint_filepath = self._get_ml_paths()
        self.mp_glob_mql_basepath, self.mp_glob_mql_data_path = self._get_mql_paths()
        
        self.DEFAULT_PARAMS = {
            'mp_glob_pl_platform_type': self.mp_glob_pl_platform_type,
            'mp_glob_pl_default_platform': self.mp_glob_pl_default_platform,
            'mp_glob_data_path': self.mp_glob_data_path,
            'mp_glob_ml_def_base_path': self.mp_glob_ml_def_base_path,
            'mp_glob_ml_num_base_path': self.mp_glob_ml_num_base_path,
            'mp_glob_ml_checkpoint_filepath': self.mp_glob_ml_checkpoint_filepath,
            'mp_glob_mql_basepath': self.mp_glob_mql_basepath,
            'mp_glob_mql_data_path': self.mp_glob_mql_data_path
        }
        
        self.params = self.DEFAULT_PARAMS
        logger.info("CMqlEnvBaseParams initialized successfully.")       
    
    def _get_data_path(self):
        """Retrieve the main data path from configuration settings."""
        base_dir = config.get('mp_glob_dir1', '8.0 Projects')
        if not base_dir:
            logger.warning("Missing 'mp_glob_dir1' in config. Using default: '8.0 Projects'.")

        project_dir = config.get('mp_glob_dir2', '8.3 ProjectModelsEquinox')
        if not project_dir:
            logger.warning("Missing 'mp_glob_dir2' in config. Using default: '8.3 ProjectModelsEquinox'.")

        platform_dir = self.platform_dir
        data_subdir = config.get('mp_glob_data_subdir', 'Mql5Data')

        home_dir = Path.home()
        one_drive = config.get('mp_glob_onedrive', 'OneDrive')
        if not one_drive:
               logger.warning("Missing 'mp_glob_onedrive' in config. Using default: 'OneDrive'.")

        data_path = home_dir / one_drive / base_dir / project_dir / platform_dir / data_subdir
        
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data path resolved: {data_path}")

        return data_path
      
    def _get_ml_paths(self):
        """Set up paths for machine learning files."""
        base_path = self.mp_glob_data_path.parent
        ml_def_path = base_path / self.platform_dir
        ml_num_path = ml_def_path / "models"
        ml_checkpoint_path = ml_def_path / "checkpoints"
        

        # Ensure directories exist
        ml_def_path.mkdir(parents=True, exist_ok=True)
        ml_num_path.mkdir(parents=True, exist_ok=True)
        ml_checkpoint_path.mkdir(parents=True, exist_ok=True)

        return ml_def_path, ml_num_path, ml_checkpoint_path
      
    def _get_mql_paths(self):
        """Configure paths for MQL-specific directories."""
        mql_dir = "Mql5"
        mql_basepath = self.mp_glob_data_path.parent / mql_dir
        mql_data_path = self.mp_glob_data_path

        # Ensure directories exist
        mql_basepath.mkdir(parents=True, exist_ok=True)
        mql_data_path.mkdir(parents=True, exist_ok=True)

        return mql_basepath, mql_data_path
