import sys
import platform
from pathlib import Path
import os
import yaml  # For loading configurations

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

from tsMqlDataParams import CMqlEnvData
from tsMqlMLParams import CMqlEnvML
from tsMqlTuneMLParams import CMqlEnvTuneML , CBaseTuneModel,CMdtunerHyperModel

class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.config = config  # Store config for later use
        self.config_platform = config.get("default_platform", None) # equinrun for Windows, equinrunlin for Linux, equinrunmac for MacOS
        logger.info(f"Setting up global environment for platform: {self.config_platform}")
         # Set the platform base
        self.mp_pl_platform_base = self.config_platform
        logger.info(f"Platform base: {self.mp_pl_platform_base}")


        # Retrieve configurations with possibility to override via kwargs
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', None)
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', None)
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', None)
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', None)

        # Use pathlib for cross-platform path handling
        # Set the home directory
        self.home_dir = Path.home()

        self.drive = kwargs.get('drive', config.get('drive', self.home_dir.anchor))
        self.user = kwargs.get('user', config.get('user', self.home_dir.name))
        self.dir1 = kwargs.get('dir1', config.get('dir1', '8.0 Projects'))
        self.dir2 = kwargs.get('dir2', config.get('dir2', '8.3 ProjectModelsEquinox'))
        self.platform_dir = kwargs.get('platform_dir', config.get('platform_dir', self.mp_pl_platform_base))
        # set the datdir
        self.datadir=kwargs.get('datadir', config.get('datadir', 'Mql5Data')) # Override if needed

        # Machine learning platform directory
        self.mp_ml_src_lib = kwargs.get('mp_ml_src_lib', config.get('mp_ml_src_lib', 'PythonLib')) # Machine learning source library
        self.mp_ml_src_modeldata = kwargs.get('mp_ml_src_modeldata', config.get('mp_ml_src_modeldata', 'tsModelData'))
        self.ml_directory = kwargs.get('ml_directory', config.get('ml_directory', 'tshybrid_ensemble_tuning_prod')) # Machine learning directory
        self.ml_model_name = kwargs.get('ml_project_name', config.get('ml_project_name', 'prjEquinox1_prod.keras')) # Machine learning project name
        self.ml_baseuniq = kwargs.get('ml_baseuniq', config.get('ml_baseuniq', '1')) # Unique identifier as string

       # data directory
        self.mp_data_dir = kwargs.get('mp_data_dir', config.get('data_dir', 'Data'))
        
       # Construct the data path using pathlib
        self.mp_data_path = Path(self.drive) / self.user / self.dir1 / self.dir2 / self.platform_dir / self.mp_data_dir 
        logger.info(f"Data path: {self.mp_data_path}")

        self.modeldatapath = self.mp_data_path

        # ML file paths, use configurations from config.yaml
        self.mp_ml_src_base = Path(self.drive) / self.user / self.dir1 / self.dir2
        self.mp_ml_src_lib = Path(config.get('mp_ml_src_lib', 'PythonLib'))
        self.mp_ml_src_modeldata = Path(config.get('mp_ml_src_modeldata', 'tsModelData'))
        self.mp_ml_directory = Path(config.get('mp_ml_directory', 'tshybrid_ensemble_tuning_prod'))
        self.mp_ml_project_name = Path(config.get('mp_ml_project_name', 'prjEquinox1_prod.keras'))
        self.mp_ml_baseuniq = config.get('mp_ml_baseuniq', '1')  # Unique identifier as string

        logger.info(f"TuneParams: Source base: {self.mp_ml_src_base}")
        logger.info(f"TuneParams: Platform base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: Source lib: {self.mp_ml_src_lib}")
        logger.info(f"TuneParams: Source data: {self.mp_ml_src_modeldata}")
        logger.info(f"TuneParams: Directory: {self.mp_ml_directory}")
        logger.info(f"TuneParams: Project name: {self.mp_ml_project_name}")
        logger.info(f"TuneParams: Baseuniq: {self.mp_ml_baseuniq}")

        self.mp_ml_def_base_path = self.mp_ml_src_base / self.mp_pl_platform_base / self.mp_ml_src_lib / self.mp_ml_src_modeldata / self.mp_ml_directory
        self.mp_ml_num_base_path = self.mp_ml_def_base_path / self.mp_ml_baseuniq
        self.mp_ml_checkpoint_filepath = self.mp_ml_def_base_path

        logger.info(f"Default base path: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path: {self.mp_ml_num_base_path}")
        logger.info(f"Checkpoint file path: {self.mp_ml_checkpoint_filepath}")

        # MQL base path
        self.mp_mql_dir1 = config.get('mp_mql_dir1', 'Mql5') 
        
        self.mp_mql_basepath = Path(self.drive) / self.user / self.dir1 / self.dir2 / self.platform_dir / self.mp_mql_dir1
        self.mp_mql_data_path = self.mp_data_path
        logger.info(f"MQL base path: {self.mp_mql_basepath}")
        logger.info(f"MQL data path: {self.mp_mql_data_path}")


    def get_params(self):
        return self.__dict__

    def run_service(self):
        # Correctly assign global environment
        self.gen_environments = {
            "globalenv": self
        }

        # Initialize data environment
        self.data_environments = {
            "dataenv": CMqlEnvData(
            )
        }

        # Initialize machine learning environment
        self.mlearn_environments = {
            "mlenv": CMqlEnvML()
        }

        # Initialize tuner environment
        self.tuner_environments = {
            "tuneenv": CMqlEnvTuneML()
        }

        # Initialize model environment
        self.model_environments = {
            "modelenv": CMdtunerHyperModel()
        }

        # Retrieve parameters safely
        genparams = {name: env.get_params() for name, env in self.gen_environments.items()}
        dataparams = {name: env.get_params() for name, env in self.data_environments.items()}
        mlearnparams = {name: env.get_params() for name, env in self.mlearn_environments.items()}
        tunerparams = {name: env.get_params() for name, env in self.tuner_environments.items()}
        modelparams = {name: env.get_params() for name, env in self.model_environments.items()}

       # Return a consolidated dictionary -  GOOD PRACTICE to return the individual params
        return {
            'genparams': genparams,
            'dataparams': dataparams,
            'mlearnparams': mlearnparams,
            'tunerparams': tunerparams,
            'modelparams': modelparams
        }

# Create a global instance
global_setter = CMqlEnvGlobal()
all_params = global_setter.run_service()  # Call run_service only ONCE.

