from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

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

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
# -- end of logging setup ----


class CMdtunerSelector:
    def __init__(self, tuner_type, backend, oracle_client, train_dataset=None, val_dataset=None, input_shape=None, num_classes=None, project_name="default_project", max_trials=10):
        self.tuner_type = tuner_type
        self.backend = backend
        self.oracle_client = oracle_client # This is the OracleClient instance
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuneobj = None # Will hold CMdtuner or PyTorchTuner instance
        self.best_model = None # To store the best model after search

        # Initialize the appropriate tuner based on backend
        if self.backend == 'tensorflow':
            self.tuneobj = CMdtuner(
                oracle_client=self.oracle_client, # Pass the oracle_client to CMdtuner
                objective="val_loss",
                max_trials=self.max_trials,
                directory=str(global_logdir),
                project_name=self.project_name,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                tuner_type=self.tuner_type,
            )
        elif self.backend == 'pytorch':
            self.tuneobj = PyTorchTuner(
                oracle_client=self.oracle_client, # Pass the oracle_client to PyTorchTuner
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        logger.info(f"CMdtunerSelector initialized for backend: {self.backend} with tuner type: {self.tuner_type}")

    def run_search(self, *args, **kwargs):
        logger.info("Chief running tuner search...")
        try:
            if self.backend == 'tensorflow':
                # For TensorFlow, the CMdtuner (KerasTuner-based) handles the search internally
                # and interacts with the OracleClient.
                logger.info("Running custom tuner search via OracleClient...")
                self.tuneobj.run_distributed_search(
                    self.train_dataset,
                    self.val_dataset,
                    epochs=kwargs.get('epochs', 10), # Default epochs if not provided
                    tuner_id="chief" # This identifies the chief
                )
                logger.info("✅ Custom tuner search completed.")

            elif self.backend == 'pytorch':
                # For PyTorch, the PyTorchTuner handles its own search logic
                logger.info("Running PyTorch tuner search...")
                self.tuneobj.run_distributed_search(tuner_id="chief")
                logger.info("✅ PyTorch tuner search completed.")

            # After the search, the chief needs to fetch the best trial from the OracleServer
            # via the OracleClient.
            logger.info("OracleClient detected in CMdtunerSelector; requesting best trials from OracleServer.")
            # Corrected line: Call get_best_trial (singular) as it exists in OracleClient
            best_trial_data = self.oracle_client.get_best_trial()

            if best_trial_data:
                logger.info(f"🏆 Best trial found: {best_trial_data.get('trial_id')} with score: {best_trial_data.get('score')}")
                # In a distributed setup, the best model is often re-built or loaded
                # based on the hyperparameters of the best trial.
                best_hyperparameters = best_trial_data.get('hyperparameters', {})
                logger.info(f"Attempting to build best model from hyperparameters: {best_hyperparameters}")
                self.best_model = self.build_model_from_hyperparameters(best_hyperparameters)
                if self.best_model:
                    logger.info("✅ Best model determined. Proceeding with export (if applicable).")
                    # This should handle saving the model found/rebuilt.
                    self.export_best_model() # Call export on self for consistency
                return self.best_model
            else:
                logger.error("❌ No best trial found. Skipping training/export.")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to fetch or finalize best trial: {e}", exc_info=True)
            return None


    def export_best_model(self, ftype='tf'):
        # This method should now ensure it exports the `best_model` that was identified
        # in `run_search` (if `build_model_from_hyperparameters` was called).
        # You might need to store `self.best_model` after it's built/loaded in run_search.
        if hasattr(self.tuneobj, 'export_best_model'):
            logger.info(f"Exporting best model using {self.tuneobj.__class__.__name__}'s export_best_model.")
            return self.tuneobj.export_best_model(ftype=ftype)
        else:
            logger.warning(f"tuneobj ({self.tuneobj.__class__.__name__}) does not have 'export_best_model' method.")
        return None

    def check_and_load_model(self, *args, **kwargs):
        if hasattr(self.tuneobj, 'check_and_load_model'):
            logger.info(f"Checking and loading model using {self.tuneobj.__class__.__name__}'s check_and_load_model.")
            return self.tuneobj.check_and_load_model(*args, **kwargs)
        logger.warning(f"tuneobj ({self.tuneobj.__class__.__name__}) does not have 'check_and_load_model' method.")
        return None

    def build_model_from_hyperparameters(self, hp):
        """
        Rebuilds the model using the given hyperparameters.
        This is crucial for the chief to get the final best model after the search.
        """
        if self.backend == 'tensorflow':
            # Assuming CMdtuner has a method to build a model from HPs
            if hasattr(self.tuneobj, 'build_model_from_hyperparameters'):
                logger.info("Building TensorFlow model from best hyperparameters.")
                return self.tuneobj.build_model_from_hyperparameters(hp)
            else:
                logger.error("CMdtuner does not have 'build_model_from_hyperparameters' method.")
                return None
        elif self.backend == 'pytorch':
            # Assuming PyTorchTuner has a method to build a model from HPs
            if hasattr(self.tuneobj, 'build_model_from_hyperparameters'):
                logger.info("Building PyTorch model from best hyperparameters.")
                return self.tuneobj.build_model_from_hyperparameters(hp)
            else:
                logger.error("PyTorchTuner does not have 'build_model_from_hyperparameters' method.")
                return None
        else:
            logger.error(f"Unsupported backend for building model from hyperparameters: {self.backend}")
            return None

