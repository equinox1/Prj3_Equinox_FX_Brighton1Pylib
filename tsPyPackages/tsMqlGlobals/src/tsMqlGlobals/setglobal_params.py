import sys
import platform
import os

from .config import config
from .logger import logger

from tsMqlDataParams import CMqlEnvData
from tsMqlML import CMqlmlsetup, CMqlEnvML
from tsMqlMLTuneParams import CMdtunerHyperModel, CMqlEnvTuneML


class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', 'df_rates1')
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', 'df_rates2')
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', 'tickdata1.csv')
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', 'ratesdata1.csv')

        # Select the platform directory
        self.config_platform = config.get("default_platform", None)
        if self.config_platform == "Windows":
            self.mp_pl_platform_base = os.path.join('equinrun')
        elif self.config_platform == "Linux":
            self.mp_pl_platform_base = os.path.join('equinrunlin')
        else:
            self.config_platform = "MacOS"
            self.mp_pl_platform_base = os.path.join('equinrunmac')

        logger.info(f"TuneParams: Platform base: {self.mp_pl_platform_base}")

        self.drive = kwargs.get('drive', 'C:')  # Drive letter for Windows, or root directory for Linux
        self.user = kwargs.get('user', 'shepa')  
        self.project = kwargs.get('project', '8.0 Projects')
        self.projectmodel = kwargs.get('projectmodel', '8.3 ProjectModelsEquinox')
        self.projectmodelname = kwargs.get('projectmodelname', self.mp_pl_platform_base)  
        self.platformdir = kwargs.get('platformdir', 'Mql5Data')  

        self.mp_data_path = os.path.join(
            self.drive, self.user, self.project, self.projectmodel, 
            self.projectmodelname, self.platformdir
        )
        logger.info(f"Data path: {self.mp_data_path}")

        # ML file paths
        self.mp_ml_src_base = os.path.join(self.drive, self.user, self.project)
        self.mp_ml_src_lib = os.path.join("PythonLib")
        self.mp_ml_src_data = os.path.join("tsModelData")
        self.mp_ml_directory = os.path.join("tshybrid_ensemble_tuning_prod")
        self.mp_ml_project_name = os.path.join("prjEquinox1_prod.keras")
        self.mp_ml_baseuniq = str(1)  

        logger.info(f"TuneParams: Source base: {self.mp_ml_src_base}")
        logger.info(f"TuneParams: Platform base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: Source lib: {self.mp_ml_src_lib}")
        logger.info(f"TuneParams: Source data: {self.mp_ml_src_data}")
        logger.info(f"TuneParams: Directory: {self.mp_ml_directory}")
        logger.info(f"TuneParams: Project name: {self.mp_ml_project_name}")
        logger.info(f"TuneParams: Baseuniq: {self.mp_ml_baseuniq}")

        self.mp_ml_def_base_path = os.path.join(
            self.mp_ml_src_base, self.mp_pl_platform_base, 
            self.mp_ml_src_lib, self.mp_ml_src_data, self.mp_ml_directory
        )

        self.mp_ml_num_base_path = os.path.join(
            self.mp_ml_def_base_path, self.mp_ml_baseuniq
        )

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

        # Initialize data environments
        self.data_environments = { 
            "dataenv": CMqlEnvData(
                globalenv=self.gen_environments["globalenv"],
                mv_data_dfname1=self.mv_data_dfname1,
                mv_data_dfname2=self.mv_data_dfname2,
                mp_data_filename1=self.mp_data_filename1,
                mp_data_filename2=self.mp_data_filename2,
            )  
        }

        self.mlearn_environments = {
            "mlenv": CMqlEnvML().get_params(),
        }

        self.tuner_environments = {
            "tuneenv": CMqlEnvTuneML(globalenv=self.gen_environments["globalenv"]).get_params(),
        }

        self.model_environments = {
            "modelenv": CMdtunerHyperModel(globalenv=self.gen_environments["globalenv"],
                                             mlenv=self.mlearn_environments["mlenv"],
                                             tuneenv=self.tuner_environments["tuneenv"]
                                            ).get_params(),
        }

        # Retrieve parameters safely
        genparams = {name: env.get_params() if hasattr(env, "get_params") else env for name, env in self.gen_environments.items()}
        dataparams = {name: env.get_params() if hasattr(env, "get_params") else env for name, env in self.data_environments.items()}
        mlearnparams = {name: env.get_params() if hasattr(env, "get_params") else env for name, env in self.mlearn_environments.items()}
        tunerparams = {name: env.get_params() if hasattr(env, "get_params") else env for name, env in self.tuner_environments.items()}
        modelparams = {name: env.get_params() if hasattr(env, "get_params") else env for name, env in self.model_environments.items()}

        # Log environment parameters
        for name, params in genparams.items():
            logger.info(f"Global Environment {name}: {params}")

        for name, params in dataparams.items():
            logger.info(f"Data Environment {name}: {params}")

        for name, params in tunerparams.items():
            logger.info(f"Tuner Environment {name}: {params}")

        for name, params in modelparams.items():
            logger.info(f"Model Environment {name}: {params}")

        return (
            self.gen_environments["globalenv"],
            self.data_environments["dataenv"],
            self.mlearn_environments["mlenv"],
            self.tuner_environments["tuneenv"],
            self.model_environments["modelenv"],
        )


# Create a global instance
global_setter = CMqlEnvGlobal()
globalenv, dataenv, mlenv, tuneenv, modelenv = global_setter.run_service()