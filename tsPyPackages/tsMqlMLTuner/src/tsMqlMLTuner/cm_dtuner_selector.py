from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

import os # Ensure os is imported
import logging
# --- Logging setup ---
# This script now *only* gets a logger. The root logger is configured by multiworker_launcher.py.
# This prevents repeated "Logging initialized" messages and ensures a consistent log file.
logger = logging.getLogger(__name__)
# -- end of logging setup ----

# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

# Removed: setup_config = CMqlSetup(...)
# This instantiation is not strictly necessary in this module's scope and was causing a NameError.

# --- Global Configuration ---
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

# Retrieve global logfile path and logdir from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
GLOBAL_LOGDIR_PATH = os.environ.get('GLOBAL_LOGDIR_PATH') # Also get logdir for CustomOracle


class CMdtunerSelector:
    def __init__(self, tuner_type, backend, oracle_client, train_dataset, val_dataset, input_shape, num_classes, project_name, max_trials, hypermodel_params, test_dataset=None, overwrite=False):
        self.tuner_type = tuner_type
        self.backend = backend
        self.oracle_client = oracle_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset # Now optional
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.project_name = project_name
        self.max_trials = max_trials
        self.overwrite = overwrite # Keep this for CMdtunerSelector's own use if needed
        self.hypermodel_params = hypermodel_params
        self.tuneobj = None
        self._initialize_tuner()

    def _initialize_tuner(self):
        if self.backend == 'tensorflow':
            logger.info("Initializing TensorFlow Tuner (CMdtuner)...")
            self.tuneobj = CMdtuner(
                oracle_client=self.oracle_client,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                objective='val_loss', # Default objective for Keras Tuner
                max_trials=self.max_trials,
                directory=GLOBAL_LOGDIR_PATH, # Use the global logdir path for tuner directory
                project_name=self.project_name,
                hypermodel_params=self.hypermodel_params,
                tuner_type=self.tuner_type,
                overwrite=self.overwrite, # CMdtuner expects overwrite explicitly
                # Pass datasets during initialization of CMdtuner
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset
            )
        elif self.backend == 'pytorch':
            logger.info("Initializing PyTorch Tuner (PyTorchTuner)...")
            self.tuneobj = PyTorchTuner(
                oracle_client=self.oracle_client,
                train_dataset=self.train_dataset, # Added train_dataset
                val_dataset=self.val_dataset,     # Added val_dataset
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                project_name=self.project_name,
                max_trials=self.max_trials,
                directory=GLOBAL_LOGDIR_PATH, # Use the global logdir path for tuner directory
                hypermodel_params=self.hypermodel_params
                # Removed 'overwrite=self.overwrite' from here.
                # PyTorchTuner should retrieve this from hypermodel_params if needed.
            )
        else:
            logger.error(f"Unsupported backend for tuner: {self.backend}")
            raise ValueError(f"Unsupported backend: {self.backend}")

    def run(self):
        logger.info(f"Running tuner for backend: {self.backend}, tuner type: {self.tuner_type}")
        if self.tuneobj:
            if self.backend == 'tensorflow':
                try:
                    # CMdtuner.run() does not take train_dataset or val_dataset as arguments.
                    # It uses the datasets already set in its __init__.
                    self.tuneobj.run()
                    best_model = self.tuneobj.finalize_best_trial()
                    logger.info("TensorFlow tuner run complete.")
                    return best_model
                except Exception as e:
                    logger.error(f"❌ Error during TensorFlow tuner run: {e}", exc_info=True)
                    return None
            elif self.backend == 'pytorch':
                try:
                    self.tuneobj.run_distributed_search(
                        train_dataset=self.train_dataset,
                        val_dataset=self.val_dataset,
                        epochs=self.hypermodel_params.get('mltune', {}).get('epochs', 10),
                        batch_size=self.hypermodel_params.get('mltune', {}).get('batch_size', 32)
                    )
                    best_model = self.tuneobj.get_best_model()
                    logger.info("PyTorch tuner run complete.")
                    return best_model
                except Exception as e:
                    logger.error(f"❌ Error during PyTorch tuner run: {e}", exc_info=True)
                    return None
        logger.warning("No tuner object initialized. Skipping run.")
        return None

    def evaluate_best_model(self, model, test_dataset):
        if self.tuneobj and hasattr(self.tuneobj, 'evaluate_model'):
            logger.info(f"Evaluating best model for {self.backend} backend...")
            if self.backend == 'tensorflow':
                return self.tuneobj.evaluate_model(model, test_dataset)
            elif self.backend == 'pytorch':
                return self.tuneobj.evaluate_model(
                    model,
                    test_data=test_dataset[0],
                    test_labels=test_dataset[1]
                )
        logger.warning("Evaluation not supported or tuner object not initialized.")
        return None, {}
