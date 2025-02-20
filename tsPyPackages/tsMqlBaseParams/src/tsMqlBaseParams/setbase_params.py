import sys
import platform
import os
from pathlib import Path
import yaml  # For loading configurations
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
from tsMqlParamMgr import BaseParamManager

class CMqlEnvBaseParams(BaseParamManager):
    """
    Environment setup for MQL-based platform, handling paths and configurations.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.config = config  # Store configuration
        self.config_platform = self.config.get("default_platform", "unknown")
        
        logger.info(f"Initializing environment for platform: {self.config_platform}")
        
        # Initialize platform and state
        self.pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = self.pchk.check_mql_state()
        if self.loadmql is None:
            logger.error("Failed to retrieve MQL state - check platform detection!")
        
        logger.info(f"Running on: {self.os_platform}, MQL state: {self.loadmql}")
        
        # Setup paths
        self.mp_glob_pl_platform_base = self._get_platform()
        self.mp_glob_data_path = self._get_data_path()
        (self.mp_glob_ml_def_base_path, self.mp_glob_ml_num_base_path, 
         self.mp_glob_ml_checkpoint_filepath) = self._get_ml_paths()
        self.mp_glob_mql_basepath, self.mp_glob_mql_data_path = self._get_mql_paths()
        
        self.DEFAULT_PARAMS = {
            'mp_glob_pl_platform_base': self.mp_glob_pl_platform_base,
            'mp_glob_data_path': self.mp_glob_data_path,
            'mp_glob_ml_def_base_path': self.mp_glob_ml_def_base_path,
            'mp_glob_ml_num_base_path': self.mp_glob_ml_num_base_path,
            'mp_glob_ml_checkpoint_filepath': self.mp_glob_ml_checkpoint_filepath,
            'mp_glob_mql_basepath': self.mp_glob_mql_basepath,
            'mp_glob_mql_data_path': self.mp_glob_mql_data_path
        }
        logger.info("CMqlEnvBaseParams initialized successfully.")
    
    def _get_platform(self):
        """Retrieve platform base setting."""
        return self.config_platform or "default"
    
    def _get_data_path(self):
        """Construct the data path based on configurations."""
        home_dir = Path.home()
        data_path = home_dir / "8.0 Projects" / "8.3 ProjectModelsEquinox" / self._get_platform() / "Data"
        logger.info(f"Data path resolved: {data_path}")
        return data_path
    
    def _get_ml_paths(self):
        """Set up paths for machine learning files."""
        base_path = Path(self.mp_glob_data_path).parent
        ml_def_path = base_path / self._get_platform()
        ml_num_path = ml_def_path / "models"
        ml_checkpoint_path = ml_def_path / "checkpoints"
        return ml_def_path, ml_num_path, ml_checkpoint_path
    
    def _get_mql_paths(self):
        """Configure paths for MQL-specific directories."""
        mql_dir = "Mql5"
        mql_basepath = self.mp_glob_data_path.parent / mql_dir
        mql_data_path = self.mp_glob_data_path
        return mql_basepath, mql_data_path