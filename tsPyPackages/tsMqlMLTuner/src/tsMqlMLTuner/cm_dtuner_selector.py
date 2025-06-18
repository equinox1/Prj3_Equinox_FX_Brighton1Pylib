from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

import os # Ensure os is imported
import logging
import time # Import time for delays in worker loop
import requests # For catching connection errors
from urllib.parse import urlparse # Import urlparse for parsing URLs

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
# This will be passed to initialize_logging...


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CMdtunerSelector:
    def __init__(self, backend, tuner_id, project_name, log_dir, train_dataset, val_dataset, test_dataset, input_shape, num_classes, max_trials, overwrite, hypermodel_params, is_chief=False, oracle_url=None, oracle_directory=None):
        self.backend = backend
        self.tuner_id = tuner_id
        self.project_name = project_name
        self.log_dir = log_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset # Store test_dataset for final evaluation
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.hypermodel_params = hypermodel_params
        self.is_chief = is_chief
        self.oracle_url = oracle_url # Store oracle_url
        self.oracle_directory = oracle_directory # Store oracle_directory
        self.tuner = None
        self.oracle = None
        self.best_model = None
        self.tuning_complete = False

        self.tuner_type = tune_params.get('tunertype', 'local') # Get tuner_type here

        # Determine tuner type and initialize OracleClient if needed
        if self.is_chief:
            # Chief always initializes Oracle for both local and remote to manage trials
            # The OracleServer will be run by the chief process itself or it will connect to a remote one.
            if self.oracle_url:
                parsed_url = urlparse(self.oracle_url)
                oracle_host = parsed_url.hostname
                oracle_port = parsed_url.port
                if oracle_port is None:
                    # Default port if not specified in URL, e.g., "http://localhost"
                    oracle_port = 9000 
                self.oracle = OracleClient(host=oracle_host, port=oracle_port) # Corrected call
                logger.info(f"CMdtunerSelector: Initialized OracleClient for chief (URL: {self.oracle_url})")
            else:
                logger.error("CMdtunerSelector: Oracle URL not provided for chief. Cannot initialize OracleClient.")
                self.oracle = None
        else: # Worker process
            # Worker always connects to the Oracle Server, so initialize OracleClient
            if self.oracle_url:
                parsed_url = urlparse(self.oracle_url)
                oracle_host = parsed_url.hostname
                oracle_port = parsed_url.port
                if oracle_port is None:
                    oracle_port = 9000
                self.oracle = OracleClient(host=oracle_host, port=oracle_port) # Corrected call
                logger.info(f"CMdtunerSelector: Initialized OracleClient for worker (URL: {self.oracle_url})")
            else:
                logger.error("CMdtunerSelector: Oracle URL not provided for worker. Cannot initialize OracleClient.")
                self.oracle = None

        if self.backend == 'tensorflow':
            logger.info("Initializing TensorFlow Tuner.")
            self.tuner = CMdtuner(
                oracle=self.oracle, # Pass the initialized oracle
                hypermodel_params=self.hypermodel_params,
                objective='val_loss',
                max_trials=self.max_trials,
                directory=self.log_dir,
                project_name=self.project_name,
                tuner_id=self.tuner_id,
                tuner_type=self.tuner_type, # Pass the determined tuner_type
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                overwrite=self.overwrite # Pass overwrite here
            )
        elif self.backend == 'pytorch':
            logger.info("Initializing PyTorch Tuner.")
            self.tuner = PyTorchTuner(
                oracle=self.oracle, # Pass the initialized oracle
                hypermodel_params=self.hypermodel_params,
                objective='val_loss', # PyTorch Tuner might use a different objective name
                max_trials=self.max_trials,
                directory=self.log_dir,
                project_name=self.project_name,
                tuner_id=self.tuner_id,
                tuner_type=self.tuner_type, # Pass the determined tuner_type
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                overwrite=self.overwrite # Pass overwrite here
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        logger.info(f"CMdtunerSelector initialized for tuner_id: {self.tuner_id}, backend: {self.backend}")


    def run(self):
        logger.info(f"CMdtunerSelector: Starting run method for tuner_id: {self.tuner_id}")
        if self.is_chief:
            self._run_chief_process()
        else:
            self._run_worker_process()
        logger.info(f"CMdtunerSelector: run method finished for tuner_id: {self.tuner_id}")
        self.tuning_complete = True


    def _run_chief_process(self):
        logger.info(f"Chief {self.tuner_id} entering tuning loop.")
        if self.tuner_type == 'local' or not self.oracle:
            logger.info("Running tuning locally (or Oracle not available for remote).")
            # For local tuning, the chief *is* the tuner.
            # It will manage trials itself.
            # PATCH: Corrected call to tuner's internal KerasTuner search method
            self.tuner.tuner.search(self.train_dataset, self.val_dataset)
        else:
            logger.info("Chief is managing trials via Oracle Server.")
            # Chief will use the OracleClient to request and manage trials for workers
            # It will iterate through trials, request new ones, and report results.
            # The actual training is done by workers.
            # This loop ensures the chief continues to request trials until max_trials is reached
            # or the Oracle indicates no more trials are available.

            trial_count = 0
            while trial_count < self.max_trials:
                logger.info(f"Chief: Requesting trial {trial_count + 1}/{self.max_trials}")
                
                # Use get_trial method that includes retry logic
                trial_response = self.oracle.get_trial() # get_trial returns a dict like {"trial": trial_data} or None
                
                if trial_response is None:
                    logger.error("Chief: Failed to get trial after multiple retries. Exiting tuning loop.")
                    break
                
                trial = trial_response.get("trial")

                if trial is None:
                    logger.info("Chief: Oracle server indicated no more trials are available.")
                    break # No more trials, break the loop
                
                # Check if trial is already completed by another worker
                if trial.get("status") == "COMPLETED":
                    logger.info(f"Chief: Trial {trial.get('trial_id')} already completed. Skipping.")
                    trial_count += 1 # Still count it towards max_trials if it's a valid trial
                    continue

                logger.info(f"Chief: Received trial ID: {trial.get('trial_id')}, Hyperparameters: {trial.get('hyperparameters')}")
                
                # Here, the chief would typically assign this trial to a worker
                # In this setup, workers pull trials directly. So the chief's role
                # is to ensure the Oracle is ready and perhaps to monitor progress.
                # For a fully distributed setup, the chief might put trials into a queue
                # for workers to pick up. For this simplified chief, it just ensures
                # trials are generated up to max_trials.

                # In KerasTuner's distributed setup, the chief primarily manages the Oracle
                # and saves the best model. Workers execute the trials.
                # The search method of the tuner (CMdtuner) is what drives trial generation
                # and execution. So, the chief still calls self.tuner.search()
                # but the Oracle handles the communication.
                
                # Re-calling search here is actually what drives the chief's role in a distributed setup.
                # It will use the Oracle to coordinate.
                # PATCH: Corrected call to tuner's internal KerasTuner search method
                self.tuner.tuner.search(self.train_dataset, self.val_dataset, trial_id=trial.get('trial_id'))
                
                trial_count += 1
        
        logger.info(f"Chief {self.tuner_id}: Tuning loop completed. Total trials processed/managed: {trial_count}")
        self.best_model = self.tuner.get_best_model()
        logger.info(f"Chief {self.tuner_id}: Best model retrieved.")

    def _run_worker_process(self):
        logger.info(f"Worker {self.tuner_id} entering tuning loop.")
        if not self.oracle:
            logger.error("Worker: OracleClient not initialized. Cannot run worker process.")
            return

        worker_active = True
        while worker_active:
            try:
                # Request a trial from the Oracle Server
                trial_response = self.oracle.get_trial() # This method already handles retries and logging
                
                if trial_response is None:
                    logger.error("Worker: Failed to get trial after multiple retries. Terminating worker.")
                    worker_active = False
                    break # Exit loop if cannot get a trial after retries

                trial = trial_response.get("trial")

                if trial is None:
                    logger.info("Worker: Oracle server indicated no more trials available or current trial limit reached.")
                    worker_active = False
                    break # No more trials to process, gracefully exit

                trial_id = trial.get("trial_id")
                hyperparameters = trial.get("hyperparameters")

                if not trial_id or not hyperparameters:
                    logger.error("Worker: Received invalid trial data from Oracle. Skipping.")
                    continue

                logger.info(f"Worker {self.tuner_id}: Processing trial ID: {trial_id} with HP: {hyperparameters}")

                try:
                    # Execute the trial using the tuner's _run_single_trial method
                    # The CMdtuner's _run_single_trial needs to be adapted to be called directly by workers
                    # and take hps and datasets.
                    # Assuming CMdtuner (or PyTorchTuner) has a method to run a single trial
                    # based on provided hyperparameters and report back.
                    
                    # Instead of directly calling _run_single_trial, let the tuner's search
                    # method handle the distributed aspect if it's set up that way.
                    # Or, if this worker directly trains, it needs to instantiate the model
                    # and run it.
                    
                    # For a KerasTuner-based approach, workers typically call tuner.search
                    # with a specific trial_id if they are coordinating, or the Oracle
                    # provides the HPs and they train directly.
                    
                    # Let's assume the Oracle provides HPs, and the worker builds & trains.
                    # This is more aligned with the "worker" concept.
                    
                    # Build and compile model with current trial's hyperparameters
                    model = self.tuner.build_model(hyperparameters)
                    
                    # Train the model
                    history = model.fit(
                        self.train_dataset,
                        epochs=tune_params.get('epochs', 10), # Get epochs from config
                        validation_data=self.val_dataset,
                        callbacks=self.tuner.get_callbacks(trial_id=trial_id, hp=hyperparameters)
                    )
                    
                    # Get the validation loss as the score
                    # Assuming 'val_loss' is the objective name
                    val_loss = history.history.get(self.tuner.objective)[-1]
                    logger.info(f"Worker {self.tuner_id}: Trial {trial_id} completed with val_loss: {val_loss}")
                    
                    # Report result back to Oracle Server
                    result = {"score": val_loss, "status": "COMPLETED"}
                    self.oracle.report_trial_result(trial_id, result) #
                    self.oracle.update_trial_status(trial_id, "COMPLETED") #

                except Exception as e:
                    logger.error(f"Worker {self.tuner_id}: Error processing trial {trial_id}: {e}", exc_info=True)
                    # Report failure to Oracle Server
                    self.oracle.update_trial_status(trial_id, "FAILED") #
                    # Continue to next trial, don't exit unless it's a critical error
                    # If it's a persistent error, perhaps worker_active should become False
                    
                time.sleep(tune_params.get('worker_delay', 5)) # Add a delay to prevent busy-waiting

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Worker {self.tuner_id}: Connection to Oracle Server lost: {e}. Retrying connection...")
                time.sleep(10) # Wait longer before retrying connection
            except Exception as e:
                logger.exception(f"Worker {self.tuner_id}: Unexpected error in worker loop: {e}")
                worker_active = False # Exit on unexpected errors
        
        logger.info(f"Worker {self.tuner_id}: Exiting tuning loop.")


    def get_best_model(self):
        logger.debug("CMdtunerSelector: get_best_model called at end.")
        if self.is_chief and self.best_model:
            return self.best_model
        elif self.oracle: # Workers or chief after tuning may retrieve the best model from the oracle
            # Attempt to load the best model based on the best trial reported to the Oracle
            best_trial_info = self.oracle.get_best_trial() #
            if best_trial_info:
                logger.info(f"CMdtunerSelector: Retrieved best trial from Oracle: {best_trial_info.get('trial_id')}")
                # You would then load the model weights corresponding to this best trial
                # This requires a mechanism to save/load models by trial_id/hyperparameters
                
                # For now, let's assume the best model from the chief's local tuner (if local)
                # or from the overall tuning process has been saved and can be loaded.
                
                # If using KerasTuner, tuner.get_best_models() will handle this
                if self.tuner:
                    best_models = self.tuner.get_best_models(num_models=1)
                    if best_models:
                        return best_models[0]
            logger.warning("CMdtunerSelector: No best model found or retrieved from Oracle/Tuner.")
            return None
        else:
            logger.warning("CMdtunerSelector: Oracle or Tuner not available to get best model.")
            return None


    def evaluate_model(self, model, test_dataset):
        if self.backend == 'tensorflow':
            logger.info("Evaluating TensorFlow model on test dataset.")
            # Ensure test_dataset is in a format compatible with model.evaluate
            # If test_dataset is already a tf.data.Dataset, pass directly.
            # Otherwise, convert X_test, y_test to a dataset.
            if isinstance(test_dataset, tf.data.Dataset):
                eval_results = model.evaluate(test_dataset)
            else:
                # Assuming test_dataset is a tuple (X_test, y_test) numpy arrays
                X_test, y_test = test_dataset
                eval_results = model.evaluate(X_test, y_test)

            logger.info(f"TensorFlow model evaluation results: {eval_results}")
            return eval_results
        elif self.backend == 'pytorch':
            if self.tuner and hasattr(self.tuner, 'evaluate_model'):
                try:
                    logger.info("Evaluating PyTorch model on test dataset.")
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