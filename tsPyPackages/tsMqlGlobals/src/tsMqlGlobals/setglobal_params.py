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
from tsMqlML import CMqlmlsetup, CMqlEnvML
from tsMqlMLTuneParams import CMdtunerHyperModel, CMqlEnvTuneML


class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.config = config  # Store config for later use

        # Retrieve configurations with possibility to override via kwargs
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', 'df_rates1')
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', 'df_rates2')
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', 'tickdata1.csv')
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', 'ratesdata1.csv')

        # Determine the platform and set the base directory accordingly
        self.config_platform = config.get("default_platform", None)
        platform_map = {
            "Windows": 'equinrun',
            "Linux": 'equinrunlin',
            "Darwin": 'equinrunmac',
            "MacOS": 'equinrunmac'
        }
        self.mp_pl_platform_base = platform_map.get(self.config_platform, 'equinrun')

        logger.info(f"TuneParams: Platform base: {self.mp_pl_platform_base}")

        # Use pathlib for cross-platform path handling
        self.home_dir = Path.home()
        self.drive = kwargs.get('drive', config.get('drive', self.home_dir.anchor))
        self.user = kwargs.get('user', config.get('user', self.home_dir.name))
        self.project = kwargs.get('project', config.get('project', '8.0 Projects'))
        self.projectmodel = kwargs.get('projectmodel', config.get('projectmodel', '8.3 ProjectModelsEquinox'))
        self.projectmodelname = kwargs.get('projectmodelname', config.get('projectmodelname', self.mp_pl_platform_base))
        self.platformdir = kwargs.get('platformdir', config.get('platformdir', 'Mql5Data'))

        # Construct the data path using pathlib
        self.mp_data_path = Path(self.drive) / self.user / self.project / self.projectmodel / self.projectmodelname / self.platformdir
        logger.info(f"Data path: {self.mp_data_path}")

        # ML file paths, use configurations from config.yaml
        self.mp_ml_src_base = Path(self.drive) / self.user / self.project
        self.mp_ml_src_lib = Path(config.get('mp_ml_src_lib', 'PythonLib'))
        self.mp_ml_src_data = Path(config.get('mp_ml_src_data', 'tsModelData'))
        self.mp_ml_directory = Path(config.get('mp_ml_directory', 'tshybrid_ensemble_tuning_prod'))
        self.mp_ml_project_name = Path(config.get('mp_ml_project_name', 'prjEquinox1_prod.keras'))
        self.mp_ml_baseuniq = config.get('mp_ml_baseuniq', '1')  # Unique identifier as string

        logger.info(f"TuneParams: Source base: {self.mp_ml_src_base}")
        logger.info(f"TuneParams: Platform base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: Source lib: {self.mp_ml_src_lib}")
        logger.info(f"TuneParams: Source data: {self.mp_ml_src_data}")
        logger.info(f"TuneParams: Directory: {self.mp_ml_directory}")
        logger.info(f"TuneParams: Project name: {self.mp_ml_project_name}")
        logger.info(f"TuneParams: Baseuniq: {self.mp_ml_baseuniq}")

        self.mp_ml_def_base_path = self.mp_ml_src_base / self.mp_pl_platform_base / self.mp_ml_src_lib / self.mp_ml_src_data / self.mp_ml_directory
        self.mp_ml_num_base_path = self.mp_ml_def_base_path / self.mp_ml_baseuniq
        self.mp_ml_checkpoint_filepath = self.mp_ml_def_base_path

        logger.info(f"Default base path: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path: {self.mp_ml_num_base_path}")
        logger.info(f"Checkpoint file path: {self.mp_ml_checkpoint_filepath}")

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
                globalenv=self.gen_environments["globalenv"],
                mv_data_dfname1=self.mv_data_dfname1,
                mv_data_dfname2=self.mv_data_dfname2,
                mp_data_filename1=self.mp_data_filename1,
                mp_data_filename2=self.mp_data_filename2,
            )
        }

        # Initialize machine learning environment
        self.mlearn_environments = {
            "mlenv": CMqlEnvML()
        }

        # Initialize tuner environment
        self.tuner_environments = {
            "tuneenv": CMqlEnvTuneML(globalenv=self.gen_environments["globalenv"])
        }

        # Initialize model environment
        self.model_environments = {
            "modelenv": CMdtunerHyperModel(
                globalenv=self.gen_environments["globalenv"],
                mlenv=self.mlearn_environments["mlenv"],
                tuneenv=self.tuner_environments["tuneenv"],
            )
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

