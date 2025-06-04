#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                            tsNeuroPredictWinMql_worker.py        |
# |                        Refactored with CMdtunerSelector          |
# +------------------------------------------------------------------+

from tsMqlSetup import CMqlSetup

import os
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5

# Setup modules
from tsMqlSetup import CMqlSetup
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides
from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlMLProcess import CDMLProcess

# Distributed tuner system
from tsMqlMLTuner import OracleClient, CMdtunerSelector # Ensure OracleClient is imported

# Keras Tuner components for manual trial management
from keras_tuner.engine.trial import TrialStatus


# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNER_ID"] = "worker" # This is important for the client to identify itself

from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides

env_backend = os.environ.get("MLTUNE_BACKEND", "tensorflow")
env_gtuner = os.environ.get("GTUNER_MODEL", env_backend)

mql_overrides = CMqlOverrides()
mql_overrides.env.override_params({
    "mltune": {"backend": env_backend},
    "app": {"gtuner_model": env_gtuner}
})

app_params = mql_overrides.env.all_params().get("app", {})
gtuner_model = app_params.get('gtuner_model', 'pytorch')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
# Added xerces_port for OracleClient initialization
xerces_port = app_params.get('xerces_port', 9000)


setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)

global_logdir, global_logfile = setup_config.set_log_dir(
    logdir=None,
    logfile=xerces_logfile,
    servername=xerces_servername,
    ltuner=gtuner_model
)

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)


# --- Strategy & Platform ---
strategy = setup_config.get_computation_strategy()
platform = run_platform.RunPlatform()
logger.info("Detected platform: %s | MetaTrader5 active: %s", platform_checker.get_platform(), platform.check_mql_state())

import multiprocessing as mp

if __name__ == "__main__":
    # Force spawn method for multiprocessing to avoid issues with TensorFlow/PyTorch
    mp.set_start_method('spawn', force=True)

# --- Main Worker Routine ---
def main(logger):
    utils = CUtilities()
    overrides = CMqlOverrides()

    # Default parameter setup
    overrides.env.override_params({"app": {'mp_app_ml_hard_run': False}})
    overrides.env.override_params({"mltune": {'batch_size': 8}})
    overrides.env.override_params({"data": {'mp_data_timeframe': mt5.TIMEFRAME_H4}})

    base = overrides.env.all_params().get("base", {})
    data = overrides.env.all_params().get("data", {})
    ml = overrides.env.all_params().get("ml", {})
    mltune = overrides.env.all_params().get("mltune", {})
    app = overrides.env.all_params().get("app", {})



    timeframe = data.get("mp_data_timeframe", mt5.TIMEFRAME_H4)
    ref = CMqlRefConfig(loaded_data_type="MINUTE", required_data_type=timeframe)
    timeconst = ref.TIME_CONSTANTS[0] if isinstance(ref.TIME_CONSTANTS, list) else ref.TIME_CONSTANTS

    now = ref.get_current_time()
    utc_from = CDataLoader().set_mql_timezone(now["CURRENTYEAR"] - 1, now["CURRENTMONTH"], now["CURRENTDAY"], now["TIMEZONE"])
    utc_to   = CDataLoader().set_mql_timezone(now["CURRENTYEAR"], now["CURRENTMONTH"], now["CURRENTDAY"], now["TIMEZONE"])

    CMqlBrokerConfig(app.get("mp_app_broker")).run_mql_login()

    dataloader = CDataLoader(
        lp_utc_from=utc_from,
        lp_utc_to=utc_to,
        lp_timeframe=timeframe,
        lp_app_primary_symbol=app.get("mp_app_primary_symbol", "EURUSD"),
        lp_app_rows=data.get("mp_data_rows", 1000),
        lp_app_rowcount=data.get("mp_data_rowcount", 10000)
    )

    df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = dataloader.run_dataloader_services()
    df_rates = CDataProcess(mp_unit=timeconst["UNIT"]["SECOND"]).run_dataprocess_services(df=df_file_rates, df_name='df_file_rates')

    X_raw, y = CDMLProcess().Create_Xy_input_and_target(df_rates, back_window=24, forward_window=24,
                                                        features=[ml.get("mp_ml_input_keyfeat", "Close")])

    # Scale
    nsamples, nsteps, nfeatures = X_raw.shape
    X_flat = X_raw.reshape((nsamples * nsteps, nfeatures))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(X_raw.shape)

    if X_scaled.ndim == 4:
        X_scaled = np.squeeze(X_scaled, axis=-1)
    elif X_scaled.ndim == 2:
        X_scaled = np.expand_dims(X_scaled, axis=-1)

    input_shape = X_scaled.shape[1:]
    logger.info("Data loaded and shaped. Input shape: %s | Output: %s", input_shape, y.shape)

    return X_scaled, y, input_shape, overrides.env.all_params()

# --- Trial Execution Loop (via Selector) ---
def run_worker_loop(X, y, input_shape, hyperparams_base):
    # Retrieve server details from app_params
    app_params = mql_overrides.env.all_params().get("app", {})
    xerces_server = app_params.get('xerces_server', '192.168.1.103')
    xerces_port = app_params.get('xerces_port', 9000)
    # Initialize OracleClient for communication with OracleServer
    oracle_client = OracleClient(host=xerces_server, port=xerces_port)

    backend = os.environ.get("MLTUNE_BACKEND", "tensorflow").lower()
    gtuner_model = os.environ.get("GTUNER_MODEL", backend).lower()

    logger.info(f"Worker Using GTuner model: {gtuner_model}")
    logger.info(f"Worker Using backend: {backend}")

    # --- Worker Loop for fetching and running trials ---
    while True:
        try:
            # Request a new trial from the OracleServer
            trial_response = oracle_client.get_trial(tuner_id=os.environ["TUNER_ID"])

            if trial_response and trial_response.get('trial_id') and trial_response.get('status') == TrialStatus.RUNNING:
                trial_id = trial_response['trial_id']
                hyperparameters = trial_response['hyperparameters']
                logger.info(f"Worker received trial {trial_id} with hyperparameters: {hyperparameters}")

                # Merge base hyperparameters with trial-specific ones
                current_hyperparams = hyperparams_base.copy()
                mltune_params = current_hyperparams.setdefault('mltune', {})
                mltune_params.update(hyperparameters) # Update with trial-specific HPs
                mltune_params.update({
                    'data_input_shape': input_shape,
                    'input_shape': input_shape,
                    'input_width': mltune_params.get('input_width', 24),
                    'shift': mltune_params.get('shift', 24)
                })

                # Instantiate the tuner for this specific trial
                if backend == "pytorch":
                    from tsMqlMLTuner.tsMqlMLTunerModTorch import PyTorchTuner
                    # Pass the OracleClient directly to the tuner for internal reporting if it supports it,
                    # or prepare to report manually. For distributed, explicit reporting is safer.
                    tuner = PyTorchTuner(
                        oracle=oracle_client, # Pass client directly
                        hypermodel_params=current_hyperparams, # Use combined hyperparams
                        traindataset=(X, y),
                        valdataset=(X, y)
                    )
                    logger.info("Worker running PyTorch model for trial %s...", trial_id)
                    # The tuner should have a method to train a single trial and return its results
                    # Assuming a method like 'fit_trial' that returns metrics for the objective
                    # This is a conceptual call; you might need to adapt your PyTorchTuner
                    metrics = tuner.run_single_trial_and_get_results(hyperparameters) # This method needs to be implemented in PyTorchTuner

                elif backend == "tensorflow":
                    import tensorflow as tf
                    buffer_size = 10000
                    batch_size = mltune_params.get('batch_size', 32) # Use batch size from HPs
                    dataset = tf.data.Dataset.from_tensor_slices((X, y))
                    dataset = dataset.shuffle(buffer_size).batch(batch_size)
                    traindataset = valdataset = testdataset = dataset

                    # CMdtunerSelector needs to be adapted to run a single trial
                    tuner_selector = CMdtunerSelector(
                        oracle=oracle_client, # Pass client directly
                        hypermodel_params=current_hyperparams, # Use combined hyperparams
                        traindataset=traindataset,
                        valdataset=valdataset,
                        testdataset=testdataset,
                        castmode='float32'
                    )
                    logger.info("Worker running TensorFlow model for trial %s...", trial_id)
                    # This method needs to be implemented in CMdtunerSelector to train a single trial
                    metrics = tuner_selector.run_single_trial_and_get_results(hyperparameters) # This method needs to be implemented in CMdtunerSelector

                else:
                    raise ValueError(f"Unsupported backend: {backend}")

                # Report results back to the OracleServer
                if metrics:
                    # Assuming 'metrics' is a dictionary like {'val_loss': 0.123, 'loss': 0.05, ...}
                    oracle_client.report_results(trial_id=trial_id, results=metrics)
                    logger.info(f"Worker reported results for trial {trial_id}: {metrics}")
                else:
                    logger.warning(f"Worker failed to get metrics for trial {trial_id}. Reporting as INCOMPLETE.")
                    # Optionally, report as INCOMPLETE if no metrics
                    oracle_client.report_results(trial_id=trial_id, results={"status": TrialStatus.INVALID})

            elif trial_response and trial_response.get('status') == TrialStatus.STOPPED:
                logger.info("OracleServer indicated STOPPED status. Exiting worker loop.")
                break # Exit loop if Oracle tells to stop

            elif trial_response and trial_response.get('status') == TrialStatus.IDLE:
                logger.info("OracleServer is IDLE. No new trials available yet. Waiting...")
                time.sleep(10) # Wait a bit before asking again
            elif not trial_response:
                 logger.info("OracleServer returned no trial. No more trials or an issue occurred. Exiting.")
                 break # No more trials available or an error occurred
            else:
                logger.info(f"Unexpected trial status from OracleServer: {trial_response.get('status')}. Exiting.")
                break # Unexpected status

        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            # Potentially mark trial as failed if an error occurred during its execution
            if 'trial_id' in locals():
                try:
                    oracle_client.report_results(trial_id=trial_id, results={"status": TrialStatus.FAILED, "error": str(e)})
                    logger.info(f"Reported trial {trial_id} as FAILED due to error.")
                except Exception as report_e:
                    logger.error(f"Failed to report trial {trial_id} as FAILED: {report_e}")
            time.sleep(5) # Wait before retrying or exiting
            # Depending on error severity, you might want to break here
            break

# --- Crucial Addition: Modify CMdtunerSelector and PyTorchTuner ---
# You will NEED to modify your CMdtunerSelector and PyTorchTuner classes
# (in tsMqlMLTuner/cm_dtuner_selector.py and tsMqlMLTuner/tsMqlMLTunerModTorch.py)
# to have a method like `run_single_trial_and_get_results(hyperparameters)`.
# This method should:
# 1. Take the specific hyperparameters for the current trial.
# 2. Build and compile the model using these hyperparameters.
# 3. Train the model for one epoch or until convergence (as per your training logic).
# 4. Return the validation metric (e.g., val_loss) as a dictionary: {'val_loss': <value>}.
#    The key must match the `objective` defined in your CustomOracle.

# Example conceptual change for CMdtunerSelector (tensorflow):
# class CMdtunerSelector:
#     # ... existing init ...
#     def run_single_trial_and_get_results(self, hyperparameters):
#         # Use self.build_model with hyperparameters
#         model = self.build_model(hyperparameters)
#         model.compile(...) # Compile with appropriate loss/optimizer/metrics
#         # Train the model
#         history = model.fit(self.traindataset, validation_data=self.valdataset, epochs=hyperparameters['epochs'])
#         # Extract the objective metric
#         val_loss = history.history['val_loss'][-1] # Or whatever your objective is
#         return {'val_loss': val_loss} # Must match objective in CustomOracle

if __name__ == "__main__":
    X, y, shape, params = main(logger)
    run_worker_loop(X, y, shape, params)
    logger.info("Worker process completed.")