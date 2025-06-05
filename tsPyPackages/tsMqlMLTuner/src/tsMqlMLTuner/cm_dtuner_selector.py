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
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup ----

class CMdtunerSelector:
    def __init__(self, **kwargs):
        # Ensure that 'oracle' from kwargs is the OracleClient for the chief
        self.oracle_client = kwargs.get('oracle')
        if not isinstance(self.oracle_client, OracleClient):
            logger.warning("CMdtunerSelector initialized without a proper OracleClient. Distributed functionality might be impacted.")

        # Store backend as an instance attribute
        self.backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        logger.info(f"Tunerselector Using backend: {self.backend}")
        print(f"Tunerselector Using backend: {self.backend}") # Keep print for immediate console feedback

        # Initialize the appropriate tuner based on the backend
        # Ensure the tuner internally uses the passed oracle_client
        if self.backend == "pytorch":
            self.tuneobj = PyTorchTuner(**kwargs)
        elif self.backend == "tensorflow":
            self.tuneobj = CMdtuner(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        logger.info(f"Initialized {self.tuneobj.__class__.__name__} with parameters: {kwargs}")
        # IMPORTANT: __init__ should not return a value. Ensure no explicit return statement.


    def run(self):
        # This method likely kicks off the worker-side run_search or similar
        # For chief, run_search is more appropriate.
        return self.tuneobj.run()

    def run_search(self):
        # Using the logger defined at the module level
        logger.info("Chief running tuner search...")

        try:
            best_model = None # Initialize best_model to None

            # The actual search is handled in the backend-specific blocks below.

            if self.backend == 'tensorflow':
                logger.info("Running KerasTuner search for TensorFlow backend...")
                # Ensure the tuner object itself is initialized within CMdtuner
                if hasattr(self.tuneobj, 'tuner') and self.tuneobj.tuner is not None:
                    self.tuneobj.tuner.search(
                        self.tuneobj.traindataset, # Use train_dataset from tuneobj
                        epochs=self.tuneobj.epochs, # Use epochs from tuneobj
                        validation_data=self.tuneobj.valdataset, # Use val_dataset from tuneobj
                        callbacks=self.tuneobj.get_callbacks(), # Get callbacks from tuneobj
                        verbose=1 # Set verbose to see progress
                    )
                    
                    # After search, attempt to get the best model
                    best_models = self.tuneobj.tuner.get_best_models(num_models=1)
                    if best_models:
                        best_model = best_models[0]
                        logger.info("Best model retrieved from TensorFlow tuner.")
                    else:
                        logger.warning("No best model found from TensorFlow tuner after search.")
                else:
                    logger.error("TensorFlow tuner (self.tuneobj.tuner) was not initialized. Cannot run search.")
                    best_model = None
                
            elif self.backend == 'pytorch':
                # For PyTorch, the PyTorchTuner handles the distributed search
                logger.info("Running distributed search for PyTorch...")
                self.tuneobj.run_distributed_search(
                    self.tuneobj.train_dataset, # Pass data to PyTorchTuner's run_distributed_search
                    self.tuneobj.val_dataset,
                    epochs=self.tuneobj.epochs, # Use epochs from tuneobj
                    batch_size=self.tuneobj.batch_size # Use batch_size from tuneobj
                )
                # After distributed search, PyTorchTuner should have a way to get the best model
                best_model = self.tuneobj.get_best_model() # Assuming this method exists in PyTorchTuner
                if best_model:
                    logger.info("Best model retrieved from PyTorch tuner.")
                else:
                    logger.warning("No best model found from PyTorch tuner.")
            else:
                logger.error(f"Unsupported backend for run_search: {self.backend}")
            
            if best_model:
                logger.info("✅ Best model determined. Proceeding with export (if applicable).")
                # This should handle saving the model found/rebuilt.
                self.export_best_model() # Call export on self for consistency
            return best_model

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
