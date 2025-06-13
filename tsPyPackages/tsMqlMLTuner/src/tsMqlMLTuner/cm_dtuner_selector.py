from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
<<<<<<< HEAD
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
tune_params = mql_overrides.env.all_params().get('mltune', {})
base_params = mql_overrides.env.all_params().get("base", {})


class CMdtunerSelector:
    def __init__(
        self,
        tuner_id,
        backend,
        oracle_client,
        train_dataset,
        val_dataset,
        test_dataset,
        input_shape,
        num_classes,
        project_name,
        max_trials,
        overwrite,
        hypermodel_params,
        **kwargs # Added to capture unexpected keyword arguments
    ):
        self.tuner_id = tuner_id
        self.backend = backend
        self.oracle_client = oracle_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.project_name = project_name
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.hypermodel_params = hypermodel_params
        self.tuneobj = None # Initialize tuneobj to None

        # Log any unexpected keyword arguments
        if kwargs:
            logger.warning(f"CMdtunerSelector received unexpected keyword arguments: {kwargs}")

        # Check for necessary components for the backend
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                tf.get_logger().setLevel(logging.ERROR) # Suppress TF warnings
                logger.info("TensorFlow backend selected.")
            except ImportError:
                logger.error("TensorFlow is not installed. Cannot use TensorFlow backend.")
                self.backend = None
        elif self.backend == 'pytorch':
            try:
                import torch
                logger.info("PyTorch backend selected.")
            except ImportError:
                logger.error("PyTorch is not installed. Cannot use PyTorch backend.")
                self.backend = None
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            self.backend = None

    def _initialize_tuner(self):
        if self.backend == 'tensorflow':
            logger.info("Initializing TensorFlow Tuner (CMdtuner)...")
            return CMdtuner(
                oracle=self.oracle_client,
                objective=self.oracle_client.objective, # Pass objective
                max_trials=self.max_trials,
                directory=self.oracle_client._directory, # Use _directory from oracle client
                project_name=self.project_name,
                overwrite=self.overwrite,
                hypermodel_params=self.hypermodel_params,
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
        elif self.backend == 'pytorch':
            logger.info("Initializing PyTorch Tuner (PyTorchTuner)...")
            return PyTorchTuner(
                oracle=self.oracle_client,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                project_name=self.project_name,
                hypermodel_params=self.hypermodel_params
            )
        logger.warning("No backend selected or supported. Tuner not initialized.")
        return None

    def run(self):
        self.tuneobj = self._initialize_tuner() # Ensure tuneobj is initialized here

        if self.tuneobj is None:
            logger.error("Tuner object failed to initialize. Cannot run tuning process.")
            return None

        logger.info(f"Running tuner for backend: {self.backend}, tuner type: {self.hypermodel_params.get('mltune', {}).get('tuner_type', 'default')}")

        best_model = None
        if self.backend == 'tensorflow':
            try:
                # Assuming CMdtuner has a run_distributed_search method
                self.tuneobj.run_distributed_search(
                    train_dataset=self.train_dataset,
                    val_dataset=self.val_dataset,
                    epochs=self.hypermodel_params.get('mltune', {}).get('epochs', 10),
                    batch_size=self.hypermodel_params.get('mltune', {}).get('batch_size', 32)
                )
                best_model = self.tuneobj.get_best_model()
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
        logger.debug(f"CMdtunerSelector: Entering evaluate_best_model. self.tuneobj type: {type(self.tuneobj)}")
        logger.debug(f"CMdtunerSelector: hasattr(self.tuneobj, 'evaluate_model'): {hasattr(self.tuneobj, 'evaluate_model')}")

        if self.tuneobj and hasattr(self.tuneobj, 'evaluate_model'):
            logger.info(f"Evaluating best model for {self.backend} backend...")
            if self.backend == 'tensorflow':
                return self.tuneobj.evaluate_model(model, test_dataset)
            elif self.backend == 'pytorch':
                # PyTorchTuner's evaluate_model expects separate test_data and test_labels
                return self.tuneobj.evaluate_model(
                    model,
                    test_data=test_dataset[0],
                    test_labels=test_dataset[1]
                )
        logger.warning("Evaluation not supported or tuner object not initialized.")
        return None, {}

    def get_best_model(self):
        if self.tuneobj:
            return self.tuneobj.get_best_model()
        return None
=======
import logging
# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup ----

class CMdtunerSelector:
    def __init__(self, **kwargs):
        backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        logger.info(f"Tunerselector Using backend: {backend}")
        print(f"Tunerselector Using backend: {backend}")
        # Initialize the appropriate tuner based on the backend
        if backend == "pytorch":
            self.tuneobj = PyTorchTuner(**kwargs)
        elif backend == "tensorflow":
            self.tuneobj = CMdtuner(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        return self.tuneobj.run()

    def run_search(self):
        return self.tuneobj.run_search()

    def export_best_model(self, ftype='tf'):
        if hasattr(self.tuneobj, 'export_best_model'):
            return self.tuneobj.export_best_model(ftype=ftype)
        return None

    def check_and_load_model(self, *args, **kwargs):
        if hasattr(self.tuneobj, 'check_and_load_model'):
            return self.tuneobj.check_and_load_model(*args, **kwargs)
        return None
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
