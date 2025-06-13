#!/usr/bin/env python3
# +------------------------------------------------------------------+\
# |                            tsNeuroPredictWinMql_worker.py        |\
# |                        Refactored with CMdtunerSelector          |\
# +------------------------------------------------------------------+\

import os
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector


# Keras Tuner components for manual trial management
from keras_tuner.engine.trial import TrialStatus

# Import mixed_precision
from tensorflow.keras import mixed_precision


# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNING_DEBUG_MODE"] = "True" # Set this to True to enable additional debug logging in the CustomOracle

# --- Logging setup ---
# This script now *only* gets a logger. The root logger is configured by multiworker_launcher.py.
# This prevents repeated "Logging initialized" messages and ensures a consistent log file.
logger = logging.getLogger(__name__)
# -- end of logging setup ----


# --- Global Configuration & Logger Setup ---
# Load environment variables and app parameters using CMqlOverrides early
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
ml_params = all_params.get("ml", {})
data_params = all_params.get("data", {})

# Determine the global backend and tuner type from environment variables, if set
MLTUNE_BACKEND = os.environ.get('MLTUNE_BACKEND', app_params.get('backend', 'tensorflow')).lower()
MLTUNE_TUNER_TYPE = os.environ.get('MLTUNE_TUNER_TYPE', tune_params.get('tuner_type', 'hyperband')).lower() # Ensure this is read

logger.info(f"Using MLTUNE_BACKEND: {MLTUNE_BACKEND}")
logger.info(f"Using MLTUNE_TUNER_TYPE: {MLTUNE_TUNER_TYPE}")

# Retrieve global log file and directory paths from environment variables.
final_logdir = os.environ.get('GLOBAL_LOGDIR_PATH')
final_logfile_path = os.environ.get('GLOBAL_LOGFILE_PATH')

# Model and project naming
MODEL_NAME = mql_overrides.env.ml_model_name() # e.g., 'tshybrid_ensemble_tuning_prod'
PROJECT_ID = mql_overrides.env.ml_project_unique_id() # e.g., '1'
PROJECT_NAME = f"{MODEL_NAME}_{PROJECT_ID}" # Combined project name for KerasTuner

# Derived paths for the worker
WORKER_LOG_DIR = Path(final_logdir) / app_params.get('xerces_servername', 'default_server') / MLTUNE_BACKEND
WORKER_MODEL_DIR = mql_overrides.env.ml_project_dir() # This should resolve to model_data/tshybrid_ensemble_tuning_prod/1
WORKER_MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

logger.info(f"Worker Log Directory: {WORKER_LOG_DIR}")
logger.info(f"Worker Model Directory: {WORKER_MODEL_DIR}")

# Configure mixed precision if enabled
if mql_overrides.env.mixed_precision_enabled():
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_global_policy(policy)
    logger.info("Mixed precision policy set to mixed_bfloat16.")

# Check platform and dependencies
platform_checker()

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(all_params, logger):
    """
    Loads and preprocesses data for training and evaluation.
    This function should be robust and return consistent data shapes.
    """
    logger.info("Starting data loading and preprocessing...")
    data_loader = CDataLoader(all_params)
    data_process = CDataProcess(all_params)
    ml_process = CDMLProcess(all_params)

    df = data_loader.load_data()
    if df is None or df.empty:
        logger.error("❌ Failed to load data or dataframe is empty.")
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Apply processing steps based on configuration
    if ml_process.ml_config.run_avg_enabled():
        df = data_process.create_hl_average(df)
        df = data_process.create_sma(df)
    if ml_process.ml_config.run_returns_enabled():
        df = data_process.create_log_returns(df)

    # Scale data (features and labels)
    features = [f"R1_{feat}" for feat in app_params.get('feature_columns', ['Open', 'High', 'Low', 'Close', 'Tick_Volume', 'spread', 'Real_Volume'])]
    
    # Ensure all features exist in the DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.error(f"❌ Missing expected features in DataFrame: {missing_features}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return None, None, None, None, None, None, None, None, None, None, None, None

    X = df[features].values
    
    # Ensure y has the correct number of features for multi-output models
    # Here, we assume y has the same features as X for prediction horizon 1
    y = df[features].values 

    # Reshape X for time series (samples, timesteps, features)
    n_steps = ml_params.get('mp_ml_pasttimeperiods', 60) # Default to 60 for look-back
    n_features = len(features) # Number of features
    prediction_horizon = ml_params.get('mp_ml_predtimeperiods', 1) # Default to 1 for prediction horizon

    if len(X) < n_steps + prediction_horizon:
        logger.error(f"❌ Not enough data to create sequences. Need at least {n_steps + prediction_horizon} rows, have {len(X)}.")
        return None, None, None, None, None, None, None, None, None, None, None, None

    X_sequences = np.array([X[i:i + n_steps] for i in range(len(X) - n_steps - prediction_horizon + 1)])
    y_sequences = np.array([y[i + n_steps : i + n_steps + prediction_horizon] for i in range(len(y) - n_steps - prediction_horizon + 1)])

    # If y_sequences is (samples, 1, features), reshape to (samples, features) for common regression tasks
    # Only reshape if prediction_horizon is 1, otherwise keep (samples, horizon, features)
    if prediction_horizon == 1:
        y_sequences = y_sequences.reshape(y_sequences.shape[0], y_sequences.shape[2])

    logger.info(f"Attempting to create sequences with processed_data_df shape: {df.shape}, look_back: {n_steps}, prediction_horizon: {prediction_horizon}, features: {features}")
    logger.info(f"✅ Sequences created. X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")

    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Flatten X_sequences for scaling, then reshape back
    original_x_shape = X_sequences.shape
    X_scaled = feature_scaler.fit_transform(X_sequences.reshape(-1, original_x_shape[-1]))
    X_scaled = X_scaled.reshape(original_x_shape)
    logger.info(f"X scaled shape: {X_scaled.shape}")

    # Flatten y_sequences for scaling, then reshape back (careful with prediction_horizon > 1)
    original_y_shape = y_sequences.shape
    if prediction_horizon == 1:
        y_scaled = target_scaler.fit_transform(y_sequences)
    else:
        # If y_sequences is (samples, horizon, features), flatten to (samples * horizon, features) for scaling
        y_scaled = target_scaler.fit_transform(y_sequences.reshape(-1, original_y_shape[-1]))
        y_scaled = y_scaled.reshape(original_y_shape)
    logger.info(f"y scaled shape: {y_scaled.shape}")
    
    logger.info("✅ Data preprocessing complete. Preparing return values.")
    logger.debug(f"Returning: 6 values. Types: {[type(val) for val in [X_scaled, y_scaled, n_steps, n_features, feature_scaler, target_scaler]]}")
    logger.debug(f"X_scaled shape={X_scaled.shape}, y_scaled shape={y_scaled.shape}, n_steps={n_steps}, n_features={n_features}, feature_scaler_type={type(feature_scaler)}, target_scaler_type={type(target_scaler)}")

    return X_scaled, y_scaled, n_steps, n_features, feature_scaler, target_scaler, df, features, X_sequences, y_sequences, ml_process.ml_config.ml_input_keyfeat() # Return all necessary values

# --- Main Worker Logic ---
def main():
    logger.info("🚀 tsNeuroPredictWinMql_worker.py started.")

    # Load and preprocess data
    (X_scaled, y_scaled, n_steps, n_features, 
     feature_scaler, target_scaler, 
     raw_df, original_features, X_sequences, y_sequences, ml_input_keyfeat) = load_and_preprocess_data(all_params, logger)

    if X_scaled is None:
        logger.error("❌ Data loading or preprocessing failed. Exiting worker.")
        sys.exit(1)

    # Split data into training, validation, and test sets
    test_size_ratio = tune_params.get('test_size_ratio', 0.2) # Default to 20%
    val_size_ratio = tune_params.get('val_size_ratio', 0.5) # Default to 50% of the test set for validation

    # First split: Separate out a combined validation/test set
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size_ratio, random_state=tune_params.get('seed', 42)
    )
    logger.info(f"Initial data split: Train {X_train.shape[0]} samples, Validation/Test {X_val_test.shape[0]} samples.")

    # Second split: Divide the validation/test set into separate validation and test sets
    if val_size_ratio > 0 and X_val_test.shape[0] > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=1 - val_size_ratio, random_state=tune_params.get('seed', 42)
        )
    else:
        # If no validation set needed, use the entire val_test as test set
        X_val, y_val = None, None
        # CORRECTED: Ensure y_test is assigned correctly here. y_val should remain None.
        X_test, y_test = X_val_test, y_val_test 
        logger.info(f"Warning: val_size_ratio is 0 or X_val_test is empty. Validation set will be empty. X_test shape: {X_test.shape}")


    logger.info(f"Final data split: Train {X_train.shape[0]} samples, Validation {X_val.shape[0] if X_val is not None else 0} samples, Test {X_test.shape[0]} samples.")
    
    input_shape = (X_train.shape[1], n_features) # (timesteps, features)
    num_classes = y_train.shape[1] if y_train.ndim > 1 else 1 # Number of output features

    logger.info(f"Determined num_classes for model output: {num_classes}")

    # Convert numpy arrays to PyTorch Tensors and create DataLoader
    # Using float32 for PyTorch model inputs
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()) if X_val is not None else None
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()) if X_test is not None else None

    # Initialize OracleClient
    oracle_client = OracleClient(
        host=app_params.get('xerces_server', '192.168.1.103'),
        port=app_params.get('xerces_port', 9000),
        tuner_id=os.environ.get('TUNER_ID', 'worker'), # Get tuner_id from env, default to 'worker'
        max_retries=tune_params.get('max_retries_per_trial', 3),
        retry_delay=5
    )
    
    logger.info(f"Worker Initializing CMdtunerSelector with backend: {MLTUNE_BACKEND} and tuner type: {MLTUNE_TUNER_TYPE}")
    # Initialize CMdtunerSelector for the worker
    tuner_config = CMdtunerSelector(
        tuner_type=MLTUNE_TUNER_TYPE,
        backend=MLTUNE_BACKEND, # Pass the correctly detected MLTUNE_BACKEND
        oracle_client=oracle_client,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset, # Pass test_dataset for final evaluation
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        project_name=MODEL_NAME, # Workers also need project_name for logging/directories
        max_trials=tune_params.get('num_trials', 1), # Workers typically run one trial at a time
        overwrite=tune_params.get('overwrite', False), # ADDED: Pass the 'overwrite' argument
        hypermodel_params=all_params # Pass all_params to the tuner for configuration
    )

    # Run the worker's tuning process (which will fetch trials from Oracle)
    logger.info("Worker starting its tuning process...")
    tuner_config.run() # Call the 'run' method for workers

    logger.info("🏁 tsNeuroPredictWinMql_worker.py finished.")


if __name__ == "__main__":
    # Ensure MetaTrader5 is initialized and finalized
    if not mt5.initialize():
        # CORRECTED: Use an f-string for the log message to prevent TypeError
        logger.error(f"❌ mt5.initialize() failed, error code = {mt5.last_error()}")
        sys.exit(1)
    else:
        logger.info("✅ MetaTrader5 initialized successfully.")

    try:
        # Run the main function
        main()
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")
