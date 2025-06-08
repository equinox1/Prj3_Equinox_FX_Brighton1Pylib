from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

import os # Ensure os is imported
import logging # Ensure logging is imported

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for cm_dtuner_selector. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
# -- end of logging setup ----

# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
tune_params = mql_overrides.env.all_params().get("mltune", {})
app_params = mql_overrides.env.all_params().get("app", {})

class CMdtunerSelector:
    def __init__(self, backend='tensorflow', tuner_type='hyperband', oracle_client=None, **kwargs):
        self.backend = backend.lower()
        self.tuner_type = tuner_type.lower()
        self.oracle = oracle_client
        logger.info(f"CMdtunerSelector initialized with backend: {self.backend}, tuner_type: {self.tuner_type}")

        if self.backend == 'tensorflow':
            logger.info("Selected TensorFlow backend.")
            self.tuneobj = CMdtuner(
                tuner_type=self.tuner_type,
                oracle_client=self.oracle,
                **kwargs
            )
        elif self.backend == 'pytorch':
            logger.info("Selected PyTorch backend.")
            self.tuneobj = PyTorchTuner(
                tuner_type=self.tuner_type, # PyTorchTuner might not use this directly, but kept for consistency
                oracle_client=self.oracle,
                **kwargs
            )
        else:
            logger.error(f"Unsupported backend: {self.backend}. Falling back to TensorFlow.")
            self.backend = 'tensorflow'
            self.tuneobj = CMdtuner(
                tuner_type=self.tuner_type,
                oracle_client=self.oracle,
                **kwargs
            )

    def search(self, *args, **kwargs):
        logger.info(f"Initiating hyperparameter search for backend: {self.backend}")
        return self.tuneobj.search(*args, **kwargs)

    def get_best_models(self, num_models=1):
        logger.info(f"Retrieving top {num_models} best models for backend: {self.backend}")
        return self.tuneobj.get_best_models(num_models=num_models)

    def finalize_best_trial(self):
        logger.info(f"Finalizing best trial for backend: {self.backend}")
        try:
            if self.backend == 'tensorflow':
                best_model = self.tuneobj.finalize_best_trial()
            elif self.backend == 'pytorch':
                best_model = self.tuneobj.finalize_best_trial()
            else:
                logger.warning(f"Finalize best trial not supported for backend: {self.backend}")
                return None

            if best_model:
                logger.info(f"Best model for {self.backend} backend successfully determined. Proceeding with export (if applicable).")
                self.export_best_model(ftype=self.backend)
            else:
                logger.warning(f"No best model found for {self.backend} backend during finalization.")
            return best_model

        except Exception as e:
            logger.error(f"❌ Failed to fetch or finalize best trial: {e}", exc_info=True)
            return None

    def export_best_model(self, ftype='tf'):
        logger.info(f"Attempting to export best model using ftype: {ftype}")
        if hasattr(self.tuneobj, 'export_best_model'):
            logger.info(f"Exporting best model using {self.tuneobj.__class__.__name__}'s export_best_model.")
            return self.tuneobj.export_best_model(ftype=ftype)
        else:
            logger.warning(f"tuneobj ({self.tuneobj.__class__.__name__}) does not have 'export_best_model' method. Skipping export.")
        return None

    def check_and_load_model(self, *args, **kwargs):
        logger.info(f"Checking and loading model for backend: {self.backend}")
        if hasattr(self.tuneobj, 'check_and_load_model'):
            logger.info(f"Checking and loading model using {self.tuneobj.__class__.__name__}'s check_and_load_model.")
            return self.tuneobj.check_and_load_model(*args, **kwargs)
        logger.warning(f"tuneobj ({self.tuneobj.__class__.__name__}) does not have 'check_and_load_model' method.")
        return None
