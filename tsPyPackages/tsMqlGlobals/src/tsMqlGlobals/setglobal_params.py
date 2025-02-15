import os
import platform
from .logger import logger
from .config import config

from tsMqlDataParams import CMqlEnvData
from tsMqlML import CMqlmlsetup, CMqlEnvML
from tsMqlMLTuneParams import CMdtunerHyperModel, CMqlEnvTuneML

class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Default values (using .get with default directly)
        self.mv_data_dfname1 = kwargs.get("mv_data_dfname1", "df_rates1")
        self.mv_data_dfname2 = kwargs.get("mv_data_dfname2", "df_rates2")
        self.mp_data_filename1 = kwargs.get("mp_data_filename1", "tickdata1.csv")
        self.mp_data_filename2 = kwargs.get("mp_data_filename2", "ratesdata1.csv")

        # Platform directory (using a dictionary for lookup)
        platform_map = {"Windows": "equinrun", "Linux": "equinrunlin", "Darwin": "equinrunmac"}
        self.config_platform = config.get("default_platform", "Windows")  # Assuming config is accessible
        self.mp_pl_platform_base = platform_map.get(self.config_platform, "equinrun")

        logger.info(f"Platform base: {self.mp_pl_platform_base}")

        # Data path setup (using os.path.join consistently)
        self.drive = kwargs.get("drive", "C:")
        self.user = kwargs.get("user", "shepa")
        self.project = kwargs.get("project", "8.0 Projects")
        self.projectmodel = kwargs.get("projectmodel", "8.3 ProjectModelsEquinox")
        self.platformdir = kwargs.get("platformdir", "Mql5Data")

        self.mp_data_path = os.path.join(self.drive, self.user, self.project, self.projectmodel, self.mp_pl_platform_base, self.platformdir)
        logger.info(f"Data path: {self.mp_data_path}")

        # ML file paths (more concise and readable)
        self.mp_ml_src_base = os.path.join(self.drive, self.user, self.project)
        self.mp_ml_src_lib = "PythonLib"
        self.mp_ml_src_data = "tsModelData"
        self.mp_ml_directory = "tshybrid_ensemble_tuning_prod"
        self.mp_ml_project_name = "prjEquinox1_prod.keras"
        self.mp_ml_baseuniq = "1"

        self.mp_ml_def_base_path = os.path.join(self.mp_ml_src_base, self.mp_pl_platform_base, self.mp_ml_src_lib, self.mp_ml_src_data, self.mp_ml_directory)
        self.mp_ml_num_base_path = os.path.join(self.mp_ml_def_base_path, self.mp_ml_baseuniq)
        self.mp_ml_checkpoint_filepath = self.mp_ml_def_base_path  # No change needed here

        logger.info(f"Default base path: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path: {self.mp_ml_num_base_path}")
        logger.info(f"Checkpoint file path: {self.mp_ml_checkpoint_filepath}")

    def get_params(self):
        return self.__dict__  # Directly return the dictionary

    def run_service(self):
        """Initialize and log environment parameters. Returns a tuple of environment objects."""

        environments = {
            "globalenv": self,
            "dataenv": CMqlEnvData(
                mv_data_dfname1=self.mv_data_dfname1,
                mv_data_dfname2=self.mv_data_dfname2,
                mp_data_filename1=self.mp_data_filename1,
                mp_data_filename2=self.mp_data_filename2,
            ),
            "mlenv": CMqlEnvML(),
            "tuneenv": CMqlEnvTuneML(),
            "modelenv": CMdtunerHyperModel(),
        }

        for name, env in environments.items():
            logger.info(f"{name}: {env.get_params()}")

        return tuple(environments.values())  # Return as a tuple directly


# Create a global instance (no change needed here)
global_setter = CMqlEnvGlobal()