from .tsMqlMLTunerMod import CMdtuner, get_callbacks # Import get_callbacks
from .tsMqlMLTunerModTorch import PyTorchTuner
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # This is crucial for chief to talk to server

import os # Ensure os is imported
import logging
import time # Import time for delays in worker loop
import requests # For catching connection errors
from urllib.parse import urlparse # Import urlparse for parsing URLs
import tensorflow as tf # <--- ADDED: Import tensorflow as tf
import torch # <--- ADDED: Import torch
import numpy as np # <--- ADDED: Import numpy


# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.\
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.\
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

# --- Global Configuration ---
# Note: These global parameters are still loaded for other parts of the module
# but CMdtunerSelector will now primarily use parameters from hypermodel_params
# for its internal configuration to improve modularity.
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {}) # Kept for other module-level uses

logger = logging.getLogger(__name__)
# Extract backend for logging path - crucial
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))


class CMdtunerSelector:
    def __init__(self, backend, tuner_id, project_name, log_dir,
                 train_dataset, val_dataset, test_dataset,
                 input_shape, num_classes, max_trials, overwrite,
                 hypermodel_params, is_chief, oracle_url=None, oracle_directory=None):
        
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
        self.hypermodel_params = hypermodel_params # This contains all_params, including 'mltune' and 'app'
        self.is_chief = is_chief
        self.oracle_url = oracle_url
        self.oracle_directory = oracle_directory # Used by chief for CustomOracle

        self.tuner = None
        self.best_model = None

        # Extract tuning parameters from hypermodel_params for internal use
        _mltune_params = self.hypermodel_params.get('mltune', {})
        self.objective_name = _mltune_params.get('objective', 'val_loss')
        self.objective_direction = _mltune_params.get('objective_direction', 'min')
        self.epochs = _mltune_params.get('max_epochs', 50) # Max epochs for training one model

        # Initialize oracle_client for workers. For chief, it will be None or handled differently.
        self.oracle_client = None
        if not self.is_chief:
            self.oracle_client = OracleClient(url=self.oracle_url, tuner_id=self.tuner_id)
            logger.info(f"Worker {self.tuner_id} initialized OracleClient connected to Oracle at {self.oracle_url}")


        logger.info(f"CMdtunerSelector initialized for tuner_id: {self.tuner_id}, backend: {self.backend}")

        if self.is_chief and self.backend == "tensorflow":
            logger.info("Initializing TensorFlow Tuner (Chief process).")
            # Chief creates the tuner which manages the Oracle (local or remote)
            tuner_type = _mltune_params.get('tuner_type', 'hyperband') # Use _mltune_params here
            self.tuner = CMdtuner(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hypermodel_params=self.hypermodel_params,
                tuner_type=tuner_type,
                directory=self.log_dir, # Base directory for logs and checkpoints
                project_name=self.project_name,
                overwrite=self.overwrite
            )
            logger.info(f"Chief Tuner '{tuner_type}' initialized.")

        elif not self.is_chief and self.backend == "tensorflow":
            logger.info("Initializing TensorFlow Worker. Connecting to Oracle.")
            # Workers connect to the Oracle via OracleClient
            # self.oracle_client is already initialized above
            logger.info(f"Worker connected to Oracle at {self.oracle_url}")
            # Worker also needs a tuner instance, but it will pull trials from the Oracle
            # It needs to know how to build the model based on hyperparameters received
            self.tuner = CMdtuner( # Worker also needs to instantiate the hypermodel for building
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hypermodel_params=self.hypermodel_params,
                tuner_type=_mltune_params.get('tuner_type', 'hyperband'), # Use _mltune_params here
                directory=self.log_dir,
                project_name=self.project_name + '_worker', # Workers can have their own project_name subdir
                overwrite=False # Workers should never overwrite, they are part of an ongoing process
            )
        elif self.backend == "pytorch":
            logger.info("Initializing PyTorch Tuner.")
            self.tuner = PyTorchTuner(
                oracle_client=self.oracle_client, # Pass the OracleClient instance
                train_dataset=self.train_dataset, # Pass the train_dataset
                val_dataset=self.val_dataset,     # Pass the val_dataset
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hypermodel_params=self.hypermodel_params,
                max_trials=self.max_trials,
                project_name=self.project_name,
                log_dir=self.log_dir,
                is_chief=self.is_chief,
                oracle_url=self.oracle_url,
                overwrite=self.overwrite,
                tuner_id=self.tuner_id # Pass tuner_id here
            )
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            raise ValueError(f"Unsupported backend: {self.backend}")

    def run(self):
        if self.backend == "tensorflow":
            self._run_tensorflow_tuning()
        elif self.backend == "pytorch":
            self._run_pytorch_tuning()
        else:
            logger.error(f"Cannot run tuning for unsupported backend: {self.backend}")

    def _run_tensorflow_tuning(self):
        logger.info(f"Starting TensorFlow tuning for {self.tuner_id} (Chief: {self.is_chief}).")
        
        # Determine model_dir for callbacks
        # Keras Tuner handles trial-specific directories internally for ModelCheckpoint
        # We need a base dir for CSVLogger/TensorBoard, which is self.tuner.project_dir
        
        # Prepare callbacks. The ModelCheckpoint filepath will be relative to tuner's trial dir.
        # OracleClient is only provided to OracleSyncCallback if it's a worker or if chief wants to self-sync
        oracle_client_for_callback = self.oracle_client if not self.is_chief else None # Or pass if chief manages its own oracle (less common)
        
        # For the chief, callbacks are automatically managed by Keras Tuner's `search` method.
        # For workers, they will also create callbacks for their specific trials.
        
        # Let's adjust `get_callbacks` to take `trial_id` and be called for each trial.
        # For now, we will pass a placeholder `trial_id` if it's the chief and using default callbacks.

        # If it's a worker, it will fetch trials and then run them.
        # The `CMdtuner`'s `run_trial` or `_build_and_fit_model` would be where callbacks
        # for that specific trial are generated and passed to `model.fit`.

        try:
            if self.is_chief:
                logger.info("Chief tuning process initiated (TensorFlow).")
                # Callbacks for chief are typically passed to tuner.search directly
                # However, for distributed setup where OracleSyncCallback needs trial_id,
                # we must manage it within a custom `run_trial` or ensure OracleClient is passed
                # to a global callback that can then get the trial.
                
                # For simplicity, let's assume `CMdtuner` (the Hyperband/RandomSearch subclass)
                # handles callbacks internally via its `run_trial` method, and we don't
                # explicitly pass them to `tuner.search` from here unless they are generic
                # (like TerminateOnNaN).
                
                # The main issue is the `y` argument. The following call is correct:
                self.tuner.search(
                    self.train_dataset,
                    validation_data=self.val_dataset,
                    epochs=self.epochs,
                    callbacks=[tf.keras.callbacks.TerminateOnNaN()], # Basic common callbacks
                    verbose=2
                )
                logger.info("Chief tuning process completed (TensorFlow).")
            else:
                logger.info("Worker tuning process initiated (TensorFlow).")
                # Worker loop to fetch trials from Oracle and run them
                self._run_tensorflow_worker_loop()
        except Exception as e:
            logger.error(f"❌ Error during TensorFlow tuning: {e}", exc_info=True)


    def _run_tensorflow_worker_loop(self):
        # This logic is adapted from standard Keras Tuner distributed worker examples
        # The worker repeatedly asks the oracle for a new trial, runs it, and reports results.
        retry_interval = 5  # seconds
        max_retries = 10
        retries = 0

        while True:
            try:
                if not self.oracle_client.is_server_available():
                    logger.warning(f"Oracle server not available at {self.oracle_url}. Retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max connection retries ({max_retries}) reached. Exiting worker.")
                        break
                    continue
                
                trial_response = self.oracle_client.get_trial(self.tuner_id)

                if trial_response is None or trial_response.get("trial_id") is None:
                    logger.info("No more trials from Oracle or Oracle is done. Exiting worker.")
                    break # Exit if no trials left or Oracle signals completion

                trial_id = trial_response.get('trial_id')
                hyperparameters = trial_response.get('hyperparameters')
                status = trial_response.get('status') # This status is from Oracle's perspective

                if status == 'STOPPED':
                    logger.info(f"Trial {trial_id} was stopped by Oracle. Skipping.")
                    continue

                if not hyperparameters:
                    logger.warning(f"Received trial {trial_id} with no hyperparameters. Skipping.")
                    self.oracle_client.update_trial_status(trial_id, status="INVALID")
                    continue

                logger.info(f"Worker {self.tuner_id} running trial: {trial_id}")
                self.oracle_client.update_trial_status(trial_id, status="RUNNING")

                hp = tf.keras.src.engine.hyperparameters.HyperParameters.from_config(hyperparameters) # Corrected import path
                model = self.tuner.hypermodel.build(hp) # Build the model using the worker's tuner instance

                # Prepare callbacks for this trial
                model_dir_for_trial = os.path.join(self.tuner.project_dir, trial_id)
                os.makedirs(model_dir_for_trial, exist_ok=True) # Ensure trial directory exists

                callbacks = get_callbacks(
                    hp,
                    model_dir=model_dir_for_trial,
                    trial_id=trial_id,
                    oracle_client=self.oracle_client, # Pass oracle client to custom callback
                    objective_name=self.objective_name,
                    direction=self.objective_direction
                )

                # Fit the model: Pass tf.data.Dataset directly
                history = model.fit(
                    self.train_dataset,
                    validation_data=self.val_dataset,
                    epochs=self.epochs,
                    callbacks=callbacks,
                    verbose=0 # Make verbose=0 for workers to reduce log spam
                )

                # Get final metrics to report to Oracle
                final_logs = history.history
                # Assuming objective is val_loss for this example
                final_objective_value = final_logs[self.objective_name][-1]
                
                results_to_report = {
                    self.objective_name: float(final_objective_value),
                    'epoch': len(final_logs[self.objective_name]) # Report number of epochs run
                }
                logger.info(f"Trial {trial_id} finished with {self.objective_name}: {final_objective_value:.4f}")
                self.oracle_client.report_trial_result(trial_id, results_to_report)
                self.oracle_client.update_trial_status(trial_id, status="COMPLETED")

            except requests.exceptions.ConnectionError as ce:
                logger.error(f"Connection to Oracle server failed: {ce}. Retrying in {retry_interval}s...")
                time.sleep(retry_interval)
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max connection retries ({max_retries}) reached. Exiting worker.")
                    break
            except Exception as e:
                logger.error(f"❌ Worker {self.tuner_id} failed during trial run: {e}", exc_info=True)
                if 'trial_id' in locals() and trial_id:
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
                break # Exit worker on unhandled error

    def _run_pytorch_tuning(self):
        logger.info(f"Starting PyTorch tuning for {self.tuner_id}.")
        try:
            # Convert TensorFlow datasets to NumPy arrays for PyTorchTuner
            # Iterate over the dataset to extract all elements
            X_train_list = []
            Y_train_list = []
            for x_batch, y_batch in self.train_dataset:
                X_train_list.append(x_batch.numpy())
                Y_train_list.append(y_batch.numpy())
            X_train_np = np.concatenate(X_train_list, axis=0)
            Y_train_np = np.concatenate(Y_train_list, axis=0)

            X_val_list = []
            Y_val_list = []
            for x_batch, y_batch in self.val_dataset:
                X_val_list.append(x_batch.numpy())
                Y_val_list.append(y_batch.numpy())
            X_val_np = np.concatenate(X_val_list, axis=0)
            Y_val_np = np.concatenate(Y_val_list, axis=0)

            if self.is_chief:
                logger.info("Chief tuning process initiated (PyTorch).")
                self.tuner.fit(
                    X_train=X_train_np, # Corrected to use the concatenated NumPy array
                    Y_train=Y_train_np, # Corrected to use the concatenated NumPy array
                    X_val=X_val_np,     # Corrected to use the concatenated NumPy array
                    Y_val=Y_val_np,     # Corrected to use the concatenated NumPy array
                    epochs=self.epochs,
                    batch_size=self.tuner.batch_size # Use batch_size from PyTorchTuner's init
                )
            else:
                logger.info("Worker tuning process initiated (PyTorch).")
                self.tuner.fit(
                    X_train=X_train_np, # Corrected to use the concatenated NumPy array
                    Y_train=Y_train_np, # Corrected to use the concatenated NumPy array
                    X_val=X_val_np,     # Corrected to use the concatenated NumPy array
                    Y_val=Y_val_np,     # Corrected to use the concatenated NumPy array
                    epochs=self.epochs,
                    batch_size=self.tuner.batch_size # Use batch_size from PyTorchTuner's init
                )
            logger.info("PyTorch tuning process completed.")
        except Exception as e:
            logger.error(f"❌ Error during PyTorch tuning: {e}", exc_info=True)

    def evaluate_model(self, model_to_evaluate, test_data_for_eval):
        # This method is called by the chief process for final evaluation
        # For TensorFlow, test_data_for_eval is a tf.data.Dataset
        # For PyTorch, test_data_for_eval is a tuple (data, labels)
        if self.backend == "tensorflow":
            if model_to_evaluate and test_data_for_eval: # test_data_for_eval should be the tf.data.Dataset
                logger.info("Evaluating TensorFlow model on test dataset.")
                try:
                    # model.evaluate expects dataset, or (x, y) if numpy arrays
                    # Since test_data_for_eval is tf.data.Dataset yielding (x, y), pass it directly
                    loss, *metrics_results = model_to_evaluate.evaluate(test_data_for_eval, verbose=0)
                    eval_results = {
                        "loss": float(loss)
                    }
                    # Map other metrics if available
                    for i, metric_name in enumerate(model_to_evaluate.metrics_names[1:]): # Skip loss
                        eval_results[metric_name] = float(metrics_results[i])
                    logger.info(f"TensorFlow model evaluation results: {eval_results}")
                    return eval_results
                except Exception as e:
                    logger.error(f"❌ Error during TensorFlow model evaluation: {e}", exc_info=True)
                    return {}
            else:
                logger.warning("No TensorFlow model or test dataset available for evaluation.")
                return {}
        elif self.backend == "pytorch":
            if model_to_evaluate and test_data_for_eval and isinstance(test_data_for_eval, tuple) and len(test_data_for_eval) == 2:
                logger.info("Evaluating PyTorch model on test dataset (numpy arrays).")
                try:
                    # PyTorchTuner's evaluate_model expects separate test_data and test_labels
                    test_data, test_labels = test_data_for_eval
                    eval_results = self.tuner.evaluate_model(
                        model_to_evaluate, # Pass the actual model instance
                        test_data=test_data,
                        test_labels=test_labels
                    )
                    # eval_results is expected to be a dictionary from PyTorchTuner
                    logger.info(f"PyTorch model evaluation results: {eval_results}")
                    return eval_results # Return the dictionary directly for consistency
                except Exception as e:
                    logger.error(f"❌ Error during PyTorch model evaluation: {e}", exc_info=True)
                    return {}
            else:
                logger.warning("No PyTorch model or invalid test data format for evaluation.")
                return {}
        else:
            logger.warning("Evaluation not supported for this backend.")
            return {}

    def get_best_model(self):
        logger.debug("CMdtunerSelector: get_best_model called at end.")
        # This method is called by the chief process after its tuning is done
        if self.backend == "tensorflow":
            if self.is_chief:
                # The chief's tuner should have the best model.
                # Use CMdtuner's finalize_best_trial method
                self.best_model = self.tuner.finalize_best_trial()
                if self.best_model:
                    logger.info("Retrieved best TensorFlow model from chief tuner.")
                else:
                    logger.warning("Could not retrieve best TensorFlow model from chief tuner.")
                return self.best_model
            else:
                logger.warning("get_best_model should primarily be called by the chief in TensorFlow distributed setup.")
                return None # Workers don't typically return the best model
        elif self.backend == "pytorch":
            # PyTorch tuner handles getting its best model
            return self.tuner.get_best_model()
        else:
            return None
