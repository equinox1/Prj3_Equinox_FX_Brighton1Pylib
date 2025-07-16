# filename: tsNeuroPredictWinMql_worker.py
TUNER_ID_CHIEF = "chief" # This constant remains for reference, but worker uses its own ID
#!/usr/bin/env python3
# +------------------------------------------------------------------+\
# |                                    tsNeuroPredictWinMql_worker.py|\\\
# |                                                    Tony Shepherd |\\\
# |                                    https://www.xercescloud.co.uk |\\\
# +------------------------------------------------------------------+\
import os
import sys
import logging
import threading
import pathlib
from pathlib import Path

import json
from datetime import datetime, date
import pytz
import socket
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Onnx and tf2onnx imports (kept for imports, but conversion logic removed from worker main)
try:
    import tf2onnx
    import onnx
    from onnx import checker
    import onnxruntime as ort
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    # print("tf2onnx, onnx, or onnxruntime not installed. ONNX conversion/verification will be skipped.") # Use logger instead
    pass # Let the logger handle this in the main function

import MetaTrader5 as mt5

# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

# Custom modules
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr # Corrected import: Changed CEnvMgr to CMqlEnvMgr
from tsMqlConnect import CMqlBrokerConfig

# Import the CMdtunerSelector
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlLogService import CMLogServiceSetup

# --- Global Configuration Loading (for module-level access if needed) ---
mql_overrides = CMqlOverrides()
all_params_global = mql_overrides.env.all_params()
app_params_global = all_params_global.get("app", {})
tune_params_global = all_params_global.get('mltune', {})
base_params_global = all_params_global.get("base", {})

# Use CMqlSetup for centralized logging (initial setup for this script's own logger)
backend_for_log_global = os.environ.get('BACKEND', tune_params_global.get('backend', 'pytorch'))

# Initialize logging for this script's module-level operations
CMLogServiceSetup.initialize_logging(
    app_params=app_params_global,
    tune_params=tune_params_global,
    base_params=base_params_global,
    role_hint='worker_module_init', # A distinct role hint for the module's own logger
    loglevel=app_params_global.get('LOGLEVEL', 'INFO').upper(),
    logfile='tsneuropredict_app.log', # Main log file for the application
)
logger = logging.getLogger(__name__)
logger.info(f"Worker module-level logging initialized. Log level: {app_params_global.get('LOGLEVEL', 'INFO').upper()}")


def generate_dummy_data(num_samples=1000, num_features=10):
    """Generates dummy data for training and validation."""
    logger.info("Generating dummy data...")
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    y = np.random.rand(num_samples, 1).astype(np.float32) * 100 # Dummy regression target
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info(f"Dummy data generated: X_train_scaled.shape={X_train_scaled.shape}, y_train.shape={y_train.shape}")
    return (X_train_scaled, y_train), (X_val_scaled, y_val), num_features


def run_worker_process_task(tuner_id: str, oracle_url: str, is_chief: bool,
                            app_params: dict, tune_params: dict, base_params: dict):
    """
    Main task function for a worker process.
    This function will be called by multiprocessing.Process.
    """
    # Re-initialize logging specifically for this process's execution context
    log_level = app_params.get('LOGLEVEL', 'INFO').upper()
    CMLogServiceSetup.initialize_logging(
        app_params=app_params,
        tune_params=tune_params,
        base_params=base_params,
        role_hint=tuner_id, # Use tuner_id (e.g., 'worker_1') as role hint for unique log file
        loglevel=log_level,
        logfile=f"{tuner_id}_{app_params.get('xerces_logfile', 'tsneuropredict_app.log')}"
    )
    process_logger = logging.getLogger(__name__)
    process_logger.info(f"Worker process task started. Tuner ID: {tuner_id}, Is Chief: {is_chief}, Oracle URL: {oracle_url}")
    process_logger.info(f"Worker process LOGDIR: {base_params.get('mp_glob_base_log_path')}")

    # Ensure MetaTrader5 is initialized for this process
    process_logger.info("PARAM HEADER: MP_APP_BROKER: %s", app_params.get('mp_app_broker'))
    broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
    mqqlobj = broker_config.run_mql_login()
    if mqqlobj is True:
        process_logger.info("Successfully logged in to MetaTrader 5.")
    else:
        process_logger.error("Failed to login to MetaTrader 5. Error code: %s", mqqlobj)
        sys.exit(1) # Exit if login fails

    try:
        # Determine backend
        backend_for_log = tune_params.get('backend', 'pytorch')

        # Setup mixed precision for TensorFlow if backend is TensorFlow/Keras
        if backend_for_log == 'tensorflow' or backend_for_log == 'keras':
            mixed_precision_policy = tune_params.get('mixed_precision_policy', 'mixed_float16')
            mixed_precision.set_global_policy(mixed_precision_policy)
            process_logger.info(f"TensorFlow Mixed Precision Policy set to: {mixed_precision_policy}")

        # 1. Generate/Load Data
        (train_data_x, train_data_y), (val_data_x, val_data_y), input_dim = generate_dummy_data()
        input_shape = (input_dim,)

        # 2. Initialize Oracle Client
        oracle_client = OracleClient(oracle_url)

        # 3. Initialize CMdtunerSelector
        tuner_selector = CMdtunerSelector(
            backend=backend_for_log,
            tuner_id=tuner_id,
            oracle_client=oracle_client,
            is_chief=is_chief, # THIS IS THE CRUCIAL CHANGE FOR A WORKER (should be False)
            oracle_url=oracle_url, # Worker MUST connect to a remote Oracle
            train_data=(train_data_x, train_data_y),
            val_data=(val_data_x, val_data_y),
            input_shape=input_shape,
            # Oracle directory for worker's local KerasTuner files should be within the main LOGDIR
            oracle_directory=str(Path(base_params.get('mp_glob_base_log_path')) / f"keras_tuner_worker_data_{tuner_id}"), # Use correct base log path for worker's oracle data
            model_save_dir=Path(base_params.get('mp_glob_base_log_path')) / "tsneuromodel_1" / "saved_models", # Pass model_save_dir
            app_params=app_params, # Pass app_params for ModelCheckpoint in TunerMod
            tune_params=tune_params # Pass tune_params for tuner configuration
        )

        process_logger.info(f"Worker {tuner_id} starting its trial execution loop...")
        tuner_selector.run() # This method now contains the worker's trial fetching loop
        process_logger.info(f"Worker {tuner_id} trial execution completed.")

        # Workers do not save the best model or perform ONNX conversion.
        # That is the responsibility of the chief process.

    except Exception as e:
        process_logger.critical(f"Unhandled exception in worker process task: {e}", exc_info=True)
        sys.exit(1) # Exit with error code

    finally:
        # Ensure MetaTrader5 is shut down properly for this process
        try:
            mt5.shutdown()
            process_logger.info("MetaTrader 5 connection shut down.")
        except Exception as e:
            process_logger.warning(f"Error during MT5 shutdown in worker process: {e}")

    process_logger.info("🏁 Worker process task finished.")


if __name__ == "__main__":
    # When this script is run as a subprocess by multiprocessing.Process,
    # the code inside this block will be executed in the new process.
    # The parameters will be passed via the 'args' of multiprocessing.Process.
    # We need to parse them from sys.argv or ensure they are set as environment variables
    # if multiprocessing's 'spawn' method is used and arguments are not directly passed.
    # For simplicity and robustness with multiprocessing, it's often better to pass
    # arguments directly to the target function.

    # For now, we'll assume environment variables are still being used for simplicity
    # with the multiprocessing.Process setup, as they were with subprocess.Popen.
    # A more robust solution would involve explicit argument passing and parsing.

    # Retrieve parameters from environment variables set by the launcher
    tuner_id = os.environ.get("TUNER_ID", "worker_standalone")
    oracle_url = os.environ.get("ORACLE_URL")
    is_chief = os.environ.get("IS_CHIEF", "false").lower() == "true"

    # Re-load parameters from environment for this specific process
    mql_overrides_child = CMqlOverrides()
    all_params_child = mql_overrides_child.env.all_params()
    app_params_child = all_params_child.get("app", {})
    tune_params_child = all_params_child.get('mltune', {})
    base_params_child = all_params_child.get("base", {})

    if not oracle_url:
        logging.getLogger(__name__).critical("ORACLE_URL environment variable not set in child process. Exiting.")
        sys.exit(1)

    run_worker_process_task(tuner_id, oracle_url, is_chief,
                            app_params_child, tune_params_child, base_params_child)

