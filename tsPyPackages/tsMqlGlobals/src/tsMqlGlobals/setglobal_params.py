import sys
import platform

from .logger import logger
from .config import config
import os

class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Select the platform dir equinrun equinrunlin equinrunmac
        self.config_platform = config.get("default_platform", None)
        if self.config_platform == "Windows":
            self.mp_pl_platform_base = os.path.join('equinrun')
        elif self.config_platform == "Linux":
            self.mp_pl_platform_base = os.path.join('equinrunlin')
        else:
            self.config_platform = "MacOS"
            self.mp_pl_platform_base = os.path.join('equinrunmac')
        logger.info(f"TuneParams:Platform base: {self.mp_pl_platform_base}")       

        self.drive = kwargs.get('drive', 'C:')  # Drive letter for the data path on linux use /mnt/c
        self.user = kwargs.get('user', 'shepa')  # User name for the data path
        self.project = kwargs.get('project', '8.0 Projects')
        self.projectmodel = kwargs.get('projectmodel', '8.3 ProjectModelsEquinox')
        self.projectmodelname = kwargs.get('projectmodelname', self.mp_pl_platform_base)  # Corrected key
        self.platformdir = kwargs.get('platformdir', 'Mql5Data')  # Corrected key
        self.mp_data_path = os.path.join(self.drive, self.user, self.project, self.projectmodel, self.projectmodelname, self.platformdir)
        logger.info(f"Data path: {self.mp_data_path}")
        
        # Ml file paths
        self.mp_ml_src_base = os.path.join(self.drive, self.user, self.project)
        self.mp_ml_src_lib = os.path.join("PythonLib")
        self.mp_ml_src_data = os.path.join("tsModelData")
        self.mp_ml_directory = os.path.join("tshybrid_ensemble_tuning_prod")
        self.mp_ml_project_name = os.path.join("prjEquinox1_prod.keras")
        self.mp_ml_baseuniq = str(1)  # str(mp_random)

        logger.info(f"TuneParams:Source base: {self.mp_ml_src_base}")
        logger.info(f"TuneParams:Platform base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams:Source lib: {self.mp_ml_src_lib}")
        logger.info(f"TuneParams:Source data: {self.mp_ml_src_data}")
        logger.info(f"TuneParams:Directory: {self.mp_ml_directory}")
        logger.info(f"TuneParams:Project name: {self.mp_ml_project_name}")
        logger.info(f"TuneParams:Baseuniq: {self.mp_ml_baseuniq}")

        self.mp_ml_def_base_path = os.path.join(self.mp_ml_src_base, self.mp_pl_platform_base, self.mp_ml_src_lib, self.mp_ml_src_data, self.mp_ml_directory)
        self.mp_ml_num_base_path = os.path.join(self.mp_ml_src_base, self.mp_pl_platform_base, self.mp_ml_src_lib, self.mp_ml_src_data, self.mp_ml_directory, self.mp_ml_baseuniq)
        self.mp_ml_checkpoint_filepath = self.mp_ml_def_base_path
        
        logger.info(f"Default base path mp: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path mp: {self.mp_ml_num_base_path}")
        logger.info(f"Checkpoint file path: {self.mp_ml_checkpoint_filepath}")
       
    def get_params(self):
        return self.__dict__  # Returns all attributes as a dictionary

# Create an instance to use across imports
global_setter = CMqlEnvGlobal()
