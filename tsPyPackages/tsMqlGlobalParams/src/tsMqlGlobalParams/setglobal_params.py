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


class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # platform settings and mql checks
        self.config = config  # Store config for later use
        self.config_platform = config.get("default_platform", None)  # equinrun for Windows, equinrunlin for Linux, equinrunmac for MacOS
        logger.info(f"Setting up global environment for platform: {self.config_platform}")

        # Set platform base
        self.mp_glob_pl_platform_base = self.config_platform
        logger.info(f"Platform base: {self.mp_glob_pl_platform_base}")

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

        # Machine Learning settings
        self.mp_glob_ml_src_lib = Path(config.get('mp_glob_ml_src_lib', 'PythonLib'))
        self.mp_glob_ml_src_modeldata = Path(config.get('mp_glob_ml_src_modeldata', 'tsModelData'))
        self.mp_glob_ml_directory = Path(config.get('mp_glob_ml_directory', 'tshybrid_ensemble_tuning_prod'))
        self.mp_glob_ml_project_name = Path(config.get('mp_glob_ml_project_name', 'prjEquinox1_prod.keras'))
        self.mp_glob_ml_baseuniq = config.get('mp_glob_ml_baseuniq', '1')

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

    def get_params(self):
        """ Returns a dictionary of all environment parameters. """
        return self.__dict__

    def run_service(self):
        """ Initializes and returns environment parameters. """

        # Correctly assign global environment
        self.gen_environments = {"globalenv": self}
        self.data_environments = {"dataenv": CMqlEnvDataParams()}
        self.mlearn_environments = {"mlenv": CMqlEnvMLParams()}
        self.tuner_environments = {"tuneenv": CMqlEnvMLTunerParams()}  
   

        # Retrieve parameters safely
        genparams = {name: env.get_params() for name, env in self.gen_environments.items()}
        dataparams = {name: env.get_params() for name, env in self.data_environments.items()}
        mlearnparams = {name: env.get_params() for name, env in self.mlearn_environments.items()}
        tunerparams = {name: env.get_params() for name, env in self.tuner_environments.items()}
        modelparams = {name: env.get_params() for name, env in self.model_environments.items()}

        # Return a consolidated dictionary
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
