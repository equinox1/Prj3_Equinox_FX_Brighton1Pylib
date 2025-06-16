from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

import os # Ensure os is imported
import logging
import time # Import time for delays in worker loop
import requests # For catching connection errors

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

# Extract backend for logging path - crucial for correct log file path
# This will be passed to initialize_logging. It can also be obtained from env if passed by launcher.
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__,
    loglevel='INFO',
    # Explicitly set the logfile name to ensure consistency
    logfile='tsneuropredict_app.log',
    # Pass the determined backend so logging goes into the correct subdirectory
    backend=backend_for_log # Pass the backend to the logging setup
)




class CMdtunerSelector:
    def __init__(self,
                 backend,
                 tuner_id,
                 project_name,
                 log_dir,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 input_shape,
                 num_classes,
                 max_trials,
                 overwrite,
                 hypermodel_params,
                 is_chief=False,
                 oracle_url=None,
                 oracle_directory=None):
        
        logger.info(f"Initializing CMdtunerSelector for {backend} backend with tuner_id: {tuner_id}")
        self.backend = backend
        self.tuner_id = tuner_id
        self.project_name = project_name
        self.log_dir = log_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.hypermodel_params = hypermodel_params
        self.is_chief = is_chief
        self.tuner = None
        self.best_model = None
        self.oracle = None
        self._start_time = time.time()

        if self.is_chief:
            # Chief creates and manages the Oracle locally
            # OracleClient is initialized for the chief to allow it to directly manage trials
            # The OracleServer will then interface with this local Oracle.
            logger.info("CMdtunerSelector (Chief): Initializing OracleClient for local Oracle management.")
            self.oracle = OracleClient(
                oracle_url=oracle_url,
                tuner_id=self.tuner_id,
                is_chief=True,
                oracle_directory=oracle_directory, # Pass the directory for local Oracle persistence
                project_name=self.project_name
            )
        else:
            # Workers connect to the remote Oracle Server
            if not oracle_url:
                logger.error("CMdtunerSelector (Worker): Oracle URL not provided. Cannot connect to Oracle Server.")
                return
            logger.info(f"CMdtunerSelector (Worker): Initializing OracleClient to connect to {oracle_url}")
            self.oracle = OracleClient(oracle_url=oracle_url, tuner_id=self.tuner_id, is_chief=False)

        # Initialize the appropriate tuner (KerasTuner for TensorFlow, custom for PyTorch)
        self._initialize_tuner()

    def _initialize_tuner(self):
        logger.debug(f"CMdtunerSelector: Initializing tuner for backend: {self.backend}")
        if self.backend == 'tensorflow':
            logger.info("CMdtunerSelector: Setting up TensorFlow tuner (CMdtuner).")
            self.tuner = CMdtuner(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                objective=self.hypermodel_params['mltune'].get('objective', 'val_loss'),
                max_trials=self.max_trials,
                directory=self.log_dir,
                project_name=self.project_name,
                seed=self.hypermodel_params['mltune'].get('seed', 42),
                hypermodel_params=self.hypermodel_params,
                overwrite=self.overwrite,
                distribution_strategy=self.hypermodel_params['mltune'].get('distribution_strategy', 'auto'),
                tuner_type=self.hypermodel_params['mltune'].get('tuner_type', 'hyperband'),
                factor=self.hypermodel_params['mltune'].get('factor', 10),
                hyperband_iterations=self.hypermodel_params['mltune'].get('hyperband_iterations', 1),
                executions_per_trial=self.hypermodel_params['mltune'].get('executions_per_trial', 1),
                tuner_id=self.tuner_id, # Pass tuner_id to CMdtuner
                oracle_client=self.oracle if not self.is_chief else None # Pass OracleClient for workers
            )
        elif self.backend == 'pytorch':
            logger.info("CMdtunerSelector: Setting up PyTorch tuner (PyTorchTuner).")
            self.tuner = PyTorchTuner(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                objective=self.hypermodel_params['mltune'].get('objective', 'val_loss'),
                max_trials=self.max_trials,
                directory=self.log_dir,
                project_name=self.project_name,
                seed=self.hypermodel_params['mltune'].get('seed', 42),
                hypermodel_params=self.hypermodel_params,
                overwrite=self.overwrite,
                tuner_type=self.hypermodel_params['mltune'].get('tuner_type', 'hyperband'),
                tuner_id=self.tuner_id, # Pass tuner_id
                oracle_client=self.oracle # PyTorchTuner always uses OracleClient for trial management
            )
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            self.tuner = None

    def _setup_chief_tuner(self):
        """Sets up the KerasTuner instance for the chief process."""
        if self.backend == 'tensorflow':
            logger.info(f"Chief {self.tuner_id}: Setting up KerasTuner (TensorFlow) for chief.")
            # For chief, the tuner manages the Oracle locally
            from keras_tuner import Hyperband, RandomSearch, BayesianOptimization
            from keras_tuner.engine.objective import Objective

            tuner_type = self.hypermodel_params['mltune'].get('tuner_type', 'hyperband').lower()
            objective = self.hypermodel_params['mltune'].get('objective', 'val_loss')

            # Ensure objective is a KerasTuner Objective object
            if isinstance(objective, str):
                objective_obj = Objective(objective, direction="min" if "loss" in objective else "max")
            else:
                objective_obj = objective # Assume it's already an Objective object

            if tuner_type == 'hyperband':
                self.tuner = Hyperband(
                    self.tuner.build_model, # Pass the hypermodel build function from CMdtuner
                    objective=objective_obj,
                    max_epochs=self.hypermodel_params['mltune'].get('max_epochs', 10),
                    factor=self.hypermodel_params['mltune'].get('factor', 3),
                    hyperband_iterations=self.hypermodel_params['mltune'].get('hyperband_iterations', 1),
                    directory=self.log_dir,
                    project_name=self.project_name,
                    overwrite=self.overwrite,
                    seed=self.hypermodel_params['mltune'].get('seed', 42),
                )
            elif tuner_type == 'randomsearch':
                self.tuner = RandomSearch(
                    self.tuner.build_model,
                    objective=objective_obj,
                    max_trials=self.max_trials,
                    directory=self.log_dir,
                    project_name=self.project_name,
                    overwrite=self.overwrite,
                    seed=self.hypermodel_params['mltune'].get('seed', 42),
                )
            elif tuner_type == 'bayesianoptimization':
                self.tuner = BayesianOptimization(
                    self.tuner.build_model,
                    objective=objective_obj,
                    max_trials=self.max_trials,
                    directory=self.log_dir,
                    project_name=self.project_name,
                    overwrite=self.overwrite,
                    seed=self.hypermodel_params['mltune'].get('seed', 42),
                )
            else:
                logger.error(f"Chief {self.tuner_id}: Unsupported Keras Tuner type: {tuner_type}")
                self.tuner = None
        elif self.backend == 'pytorch':
            logger.info(f"Chief {self.tuner_id}: PyTorch backend detected. CMdtunerSelector as Chief will manage Oracle and not use KerasTuner directly for search.")
            # For PyTorch Chief, the OracleClient is used to manage trials for workers
            # The 'run' method will not call self.tuner.search for PyTorch Chief.
            pass # No specific tuner setup needed here, as OracleClient handles trial management

    def run(self):
        logger.debug(f"CMdtunerSelector: Entering run method. self.is_chief: {self.is_chief}, backend: {self.backend}")
        
        if self.is_chief:
            # Chief logic
            logger.info(f"Chief {self.tuner_id} is setting up and running the tuning process.")
            if self.backend == 'tensorflow':
                self._setup_chief_tuner() # This will initialize self.tuner for TensorFlow
                if self.tuner:
                    logger.info(f"Chief {self.tuner_id} starting TensorFlow tuner search.")
                    try:
                        self.tuner.search(self.train_dataset,
                                          epochs=self.hypermodel_params['mltune'].get('epochs', 2),
                                          validation_data=self.val_dataset,
                                          callbacks=self.tuner._get_callbacks(self.log_dir)) # Use tuner's _get_callbacks
                        logger.info(f"Chief {self.tuner_id} TensorFlow tuner search completed.")
                    except Exception as e:
                        logger.error(f"❌ Chief {self.tuner_id}: Error during TensorFlow tuner search: {e}", exc_info=True)
                else:
                    logger.error(f"❌ Chief {self.tuner_id}: KerasTuner instance not initialized for chief.")
            elif self.backend == 'pytorch':
                logger.info(f"Chief {self.tuner_id}: PyTorch backend detected. Chief will manage Oracle.")
                # Chief for PyTorch backend mainly manages the Oracle.
                # It doesn't run a 'search' loop like KerasTuner, but rather waits for workers
                # to request trials and reports results. For this simple setup,
                # the chief's run might just initialize the Oracle and then effectively wait.
                # In a more complex setup, it might monitor trials or trigger analysis.
                if self.oracle:
                    logger.info(f"Chief {self.tuner_id}: Oracle is initialized and ready to serve trials.")
                    # Keep chief alive to serve trials to workers
                    # In a real daemon, this would be a long-running process
                    # For now, let's just make it wait, assuming the launcher manages its lifecycle.
                    while True:
                        time.sleep(60) # Chief waits indefinitely, serving requests
                else:
                    logger.error(f"❌ Chief {self.tuner_id}: OracleClient not initialized for chief. Cannot serve trials.")
            else:
                logger.error(f"Chief {self.tuner_id}: Unsupported backend: {self.backend}")

        else: # Worker logic
            if not self.oracle:
                logger.error("❌ Worker: OracleClient not initialized. Cannot proceed with tuning.")
                return

            logger.info(f"Worker {self.tuner_id} entering tuning loop to fetch trials from Oracle.")
            
            completed_trials = 0
            max_worker_trials_limit = self.hypermodel_params['mltune'].get('max_trials', 1) # Workers usually run 1 trial

            # Add a safety counter for failed trial requests to prevent infinite loops on a broken Oracle
            consecutive_failed_requests = 0
            max_consecutive_failed_requests = self.hypermodel_params['mltune'].get('max_consecutive_failed_trials', 5)
            
            while completed_trials < max_worker_trials_limit:
                trial_id = None # Initialize trial_id for consistent error reporting
                try:
                    logger.info(f"Worker {self.tuner_id} requesting new trial from Oracle (attempt {consecutive_failed_requests + 1}/{max_consecutive_failed_requests}).")
                    
                    # Ensure OracleClient.request_trial handles potential network issues internally (with retries)
                    trial_info = self.oracle.request_trial(self.tuner_id)
                    
                    if trial_info and trial_info.get("trial_id"):
                        consecutive_failed_requests = 0 # Reset counter on success
                        trial_id = trial_info["trial_id"]
                        hyperparameters = trial_info["hyperparameters"]
                        logger.info(f"Worker {self.tuner_id} received trial {trial_id} "
                                    f"with hyperparameters: {hyperparameters}")

                        try:
                            if self.backend == 'tensorflow':
                                logger.info(f"Worker {self.tuner_id}: Building and training TensorFlow model for trial {trial_id}.")
                                model = self.tuner.build_model(hyperparameters) # Use self.tuner.build_model
                                
                                history = model.fit(self.train_dataset,
                                                    epochs=self.hypermodel_params['mltune'].get('epochs', 2),
                                                    validation_data=self.val_dataset,
                                                    verbose=0)

                                val_loss = history.history.get('val_loss')[-1] if 'val_loss' in history.history else None
                                metrics = {k: v[-1] for k, v in history.history.items()} if history else {}
                                
                                logger.info(f"Worker {self.tuner_id} finished TensorFlow training for trial {trial_id}. Val Loss: {val_loss}")
                                
                                self.oracle.report_trial_result(trial_id, metrics)
                                self.oracle.update_trial_status(trial_id, status="COMPLETED")
                                logger.info(f"Worker {self.tuner_id} reported result for TensorFlow trial {trial_id}.")
                                completed_trials += 1

                            elif self.backend == 'pytorch':
                                logger.info(f"Worker {self.tuner_id}: Building and training PyTorch model for trial {trial_id}.")
                                # Assuming PyTorchTuner object is self.tuner
                                if not isinstance(self.tuner, PyTorchTuner):
                                    raise TypeError("PyTorchTuner instance not correctly initialized for PyTorch backend.")

                                # The PyTorchTuner's `train_model_for_trial` handles the training loop
                                history_metrics = self.tuner.train_model_for_trial(
                                    hyperparameters,
                                    self.train_dataset,
                                    self.val_dataset,
                                    epochs=self.hypermodel_params['mltune'].get('epochs', 2),
                                    batch_size=self.hypermodel_params['mltune'].get('batch_size', 32)
                                )
                                
                                val_loss = history_metrics.get('val_loss')
                                logger.info(f"Worker {self.tuner_id} finished PyTorch training for trial {trial_id}. Val Loss: {val_loss}")
                                
                                self.oracle.report_trial_result(trial_id, history_metrics)
                                self.oracle.update_trial_status(trial_id, status="COMPLETED")
                                logger.info(f"Worker {self.tuner_id} reported result for PyTorch trial {trial_id}.")
                                completed_trials += 1

                            else:
                                logger.error(f"Worker {self.tuner_id}: Unsupported backend: {self.backend}. Marking trial {trial_id} as FAILED.")
                                self.oracle.update_trial_status(trial_id, status="FAILED")
                                break # Exit loop if backend is unsupported

                        except Exception as e:
                            logger.error(f"❌ Worker {self.tuner_id}: Error during model build/train for trial {trial_id}: {e}", exc_info=True)
                            if trial_id: # Only update status if a trial was actually received
                                self.oracle.update_trial_status(trial_id, status="FAILED")
                            consecutive_failed_requests += 1 # Count as a failed attempt to process trial
                            time.sleep(self.oracle.request_interval) # Wait before retrying

                    else:
                        logger.info(f"Worker {self.tuner_id} received no new trial from Oracle. Max trials might be reached or Oracle busy. Waiting...")
                        consecutive_failed_requests += 1
                        time.sleep(self.oracle.request_interval) # Wait before retrying
                        
                        if consecutive_failed_requests >= max_consecutive_failed_requests:
                            logger.warning(f"Worker {self.tuner_id} consistently failed to get trials "
                                        f"({max_consecutive_failed_requests} consecutive attempts). Exiting tuning loop.")
                            break # Exit loop if too many consecutive failed requests

                except requests.exceptions.ConnectionError as ce:
                    logger.error(f"❌ Worker {self.tuner_id} connection error to OracleServer at {self.oracle.url}: {ce}. Retrying in {self.oracle.request_interval} seconds...")
                    consecutive_failed_requests += 1
                    time.sleep(self.oracle.request_interval) # Wait before retrying
                    if consecutive_failed_requests >= max_consecutive_failed_requests:
                        logger.critical(f"❌ Worker {self.tuner_id} lost connection to OracleServer for too long. Exiting.")
                        break # Critical error, exit the loop
                except Exception as e:
                    logger.error(f"❌ Worker {self.tuner_id} encountered an unexpected error during tuning loop: {e}", exc_info=True)
                    if trial_id:
                        self.oracle.update_trial_status(trial_id, status="FAILED")
                    consecutive_failed_requests += 1
                    time.sleep(self.oracle.request_interval) # Wait before retrying
                    if consecutive_failed_requests >= max_consecutive_failed_requests:
                        logger.critical(f"❌ Worker {self.tuner_id} encountered too many consecutive errors. Exiting tuning loop.")
                        break
            logger.info(f"Worker {self.tuner_id} finished its tuning process after processing {completed_trials} trials.")

    def get_best_model(self):
        logger.debug("CMdtunerSelector: Entering get_best_model.")
        if self.is_chief and self.backend == 'tensorflow':
            if self.tuner:
                try:
                    logger.info("Chief is retrieving the best model from KerasTuner.")
                    self.best_model = self.tuner.get_best_models(num_models=1)[0]
                    return self.best_model
                except Exception as e:
                    logger.error(f"❌ Error retrieving best model from KerasTuner: {e}", exc_info=True)
                    return None
            else:
                logger.warning("KerasTuner object not initialized for chief. Cannot get best model.")
                return None
        elif self.backend == 'pytorch':
            # For PyTorch, the best model needs to be explicitly loaded using the best hyperparameters
            # retrieved from the Oracle.
            logger.info("Retrieving best PyTorch model info from Oracle.")
            if self.oracle:
                best_trial = self.oracle.get_best_trial()
                if best_trial and 'hyperparameters' in best_trial:
                    best_hps = best_trial['hyperparameters']
                    logger.info(f"Best PyTorch trial found: {best_trial.get('trial_id')}, HPs: {best_hps}, Score: {best_trial.get('score')}")
                    # Assume PyTorchTuner has a method to load/rebuild the best model
                    if self.tuner and hasattr(self.tuner, 'load_best_model_from_oracle'):
                        try:
                            # Pass the oracle client and potentially the specific trial info
                            self.best_model = self.tuner.load_best_model_from_oracle(best_hps)
                            return self.best_model
                        except Exception as e:
                            logger.error(f"Error loading best PyTorch model via tuner: {e}", exc_info=True)
                            return None
                    else:
                        logger.warning("PyTorchTuner or its 'load_best_model_from_oracle' method not available.")
                        return None
                else:
                    logger.warning("No best trial information available from Oracle for PyTorch backend.")
                    return None
            else:
                logger.warning("OracleClient not initialized. Cannot get best PyTorch model.")
                return None
        else:
            logger.warning(f"Unsupported backend '{self.backend}' for get_best_model operation.")
            return None

    def evaluate_best_model(self, model, test_dataset):
        logger.debug(f"CMdtunerSelector: Entering evaluate_best_model. self.backend: {self.backend}")
        
        if self.backend == 'tensorflow':
            if not model:
                logger.warning("No TensorFlow model provided for evaluation.")
                return None, {}
            logger.info("Evaluating best TensorFlow model...")
            # Assuming test_dataset is a tf.data.Dataset
            try:
                loss, *metrics = model.evaluate(test_dataset, verbose=0)
                # Keras model.evaluate returns loss and then other metrics in order
                metric_names = model.metrics_names
                results = dict(zip(metric_names, [loss] + metrics))
                logger.info(f"TensorFlow model evaluation results: {results}")
                return loss, results
            except Exception as e:
                logger.error(f"❌ Error during TensorFlow model evaluation: {e}", exc_info=True)
                return None, {}
        elif self.backend == 'pytorch':
            if not model:
                logger.warning("No PyTorch model provided for evaluation.")
                return None, {}
            logger.info("Evaluating best PyTorch model...")
            if self.tuner and hasattr(self.tuner, 'evaluate_model'):
                # Assuming evaluate_model in PyTorchTuner handles data loading/conversion
                try:
                    # PyTorchTuner's evaluate_model expects separate test_data and test_labels
                    # Assuming test_dataset is a tuple (test_data, test_labels)
                    test_data, test_labels = test_dataset 
                    eval_results = self.tuner.evaluate_model(
                        model,
                        test_data=test_data,
                        test_labels=test_labels
                    )
                    # eval_results is expected to be a dictionary from PyTorchTuner
                    val_loss = eval_results.get('val_loss') # Or whatever the primary metric is
                    logger.info(f"PyTorch model evaluation results: {eval_results}")
                    return val_loss, eval_results
                except Exception as e:
                    logger.error(f"❌ Error during PyTorch model evaluation: {e}", exc_info=True)
                    return None, {}
            else:
                logger.warning("PyTorchTuner or its 'evaluate_model' method not available. Cannot evaluate PyTorch model.")
                return None, {}
        else:
            logger.warning("Evaluation not supported for this backend.")
            return None, {}

    def get_best_model(self):
        logger.debug("CMdtunerSelector: get_best_model called at end.")
        # This method is called by the chief process after its tuning is done
        # The logic for getting the best model is primarily in the first get_best_model.
        return self.best_model
