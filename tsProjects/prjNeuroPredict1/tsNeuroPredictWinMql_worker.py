#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                            tsNeuroPredictWinMql_worker.py        |
# |                        Refactored with CMdtunerSelector          |
# +------------------------------------------------------------------+\

import os
import sys
import logging # Import logging first
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
os.environ["TUNER_ID"] = "worker" # This is important for the client to identify itself


# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

# -- Set up global logging (from tsMqlSetup) --
# This block configures the root logger, so it should run before any logger.getLogger(__name__) calls
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for Worker. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module AFTER setup_logging
# -- end of logging setup ----


gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
logger.info(f"Worker get: Using backend: {backend} and tuner type: {gtuner_model}")
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')


# --- Set mixed precision policy if using TensorFlow backend ---
if backend == 'tensorflow':
    try:
        mixed_precision.set_global_policy('mixed_bfloat16')
        logger.info("✅ TensorFlow mixed precision policy set to 'mixed_bfloat16'.")
    except Exception as e:
        logger.warning(f"⚠️ Could not set mixed precision policy: {e}. Falling back to default float type.")


# --- Global Configuration ---
# Use CMqlEnvMgr to get all parameters
env_mgr = CMqlEnvMgr()
all_params = env_mgr.all_params()
base_params = all_params.get("base", {})
app_params = all_params.get("app", {})
broker_params = all_params.get("broker", {})
ml_params = all_params.get("ml", {})
tune_params = all_params.get("mltune", {})

# Extract necessary parameters for Worker
SYMBOLS = app_params.get('mp_app_symbols', ['EURUSD'])
TIMEFRAME = app_params.get('mp_app_timeframe', 'M1')
NUM_CANDLES = app_params.get('mp_app_num_candles', 10000)
MODEL_NAME = app_params.get('mp_glob_sub_ml_model_name', 'ts_mql_model')
LOOK_BACK = ml_params.get('mp_ml_look_back', 60)
PREDICTION_HORIZON = ml_params.get('mp_ml_prediction_horizon', 1)
TRAIN_SPLIT_RATIO = ml_params.get('mp_ml_train_split_ratio', 0.8)
FEATURES_TO_USE = ml_params.get('mp_ml_features_to_use', ['R1_Open', 'R1_High', 'R1_Low', 'R1_Close', 'R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume'])
TARGET_FEATURE = ml_params.get('mp_ml_target_feature', 'R1_Close')
NORMALIZATION_METHOD = ml_params.get('mp_ml_normalization_method', 'StandardScaler')
MLTUNE_BACKEND = backend
MLTUNE_TUNER_TYPE = gtuner_model
ORACLE_HOST = app_params.get('xerces_server', '192.168.1.103')
ORACLE_PORT = app_params.get('xerces_port', 9000)


# Mapping for MetaTrader5 timeframes
MT5_TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M2': mt5.TIMEFRAME_M2,
    'M3': mt5.TIMEFRAME_M3,
    'M4': mt5.TIMEFRAME_M4,
    'M5': mt5.TIMEFRAME_M5,
    'M6': mt5.TIMEFRAME_M6,
    'M10': mt5.TIMEFRAME_M10,
    'M12': mt5.TIMEFRAME_M12,
    'M15': mt5.TIMEFRAME_M15,
    'M20': mt5.TIMEFRAME_M20,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H2': mt5.TIMEFRAME_H2,
    'H3': mt5.TIMEFRAME_H3,
    'H4': mt5.TIMEFRAME_H4,
    'H6': mt5.TIMEFRAME_H6,
    'H8': mt5.TIMEFRAME_H8,
    'H12': mt5.TIMEFRAME_H12,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(symbol, timeframe_str, num_candles, look_back, prediction_horizon, features_to_use, target_feature, normalization_method):
    logger.info(f"📊 Loading data for {symbol} {timeframe_str}...")
    
    # Convert string timeframe to mt5.TIMEFRAME_* constant
    timeframe_mt5 = MT5_TIMEFRAME_MAP.get(timeframe_str)
    if timeframe_mt5 is None:
        logger.error(f"❌ Invalid timeframe string: {timeframe_str}. Please use one of: {list(MT5_TIMEFRAME_MAP.keys())}")
        return None, None, None, None, None, None

    # Initialize CDataLoader with the correct mt5 timeframe constant
    data_loader = CDataLoader(
        lp_app_primary_symbol=symbol,
        lp_timeframe=timeframe_mt5, # Pass the mt5 constant
        lp_data_rows=num_candles # Assuming num_candles corresponds to lp_data_rows
    )
    
    # Call run_dataloader_services to get the dataframes
    df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = data_loader.run_dataloader_services()

    # Assuming 'df_api_rates' is the primary dataframe for historical rates
    data_df = df_api_rates 

    if data_df.empty:
        logger.error(f"❌ No data loaded for {symbol}.")
        return None, None, None, None, None, None

    logger.info("⚙️ Preprocessing data (CDataProcess)...")
    # Initialize CDataProcess with keyword arguments
    data_process = CDataProcess(
        look_back=look_back,
        prediction_horizon=prediction_horizon,
        features_to_use=features_to_use,
        target_feature=target_feature
    )
    
    # Call run_dataprocess_services to process the dataframe
    processed_data_df = data_process.run_dataprocess_services(df=data_df, df_name='df_api_rates') # Pass df and df_name

    if processed_data_df.empty:
        logger.error("❌ Data processing resulted in an empty DataFrame.")
        return None, None, None, None, None, None

    logger.info("⚙️ Creating ML sequences (CDMLProcess)...")
    ml_process = CDMLProcess(
        look_back=look_back,
        prediction_horizon=prediction_horizon,
        features_to_use=features_to_use,
        target_feature=target_feature
    )

    logger.info(f"Attempting to create sequences with processed_data_df shape: {processed_data_df.shape}, look_back: {look_back}, prediction_horizon: {prediction_horizon}, features: {features_to_use}")
    X, y = ml_process.Create_Xy_input_and_target(
        df=processed_data_df,
        back_window=look_back,
        forward_window=prediction_horizon,
        features=features_to_use # Pass the list of features
    )

    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error(f"❌ Failed to create sequences after data processing. X shape: {X.shape if X is not None else 'None'}, y shape: {y.shape if y is not None else 'None'}")
        return None, None, None, None, None, None

    n_steps = X.shape[1]
    n_features = X.shape[2]
    logger.info(f"✅ Sequences created. X shape: {X.shape}, y shape: {y.shape}")

    if y.ndim == 1:
        y_reshaped_for_scaler = y.reshape(-1, 1)
    else:
        y_reshaped_for_scaler = y

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
    logger.info(f"X scaled shape: {X_scaled.shape}")

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_reshaped_for_scaler)

    if y.ndim == 1:
        y_scaled = y_scaled.flatten()
    logger.info(f"y scaled shape: {y_scaled.shape}")

    logger.info("✅ Data preprocessing complete.")
    return X_scaled, y_scaled, n_steps, n_features, feature_scaler, target_scaler


def main(logger):
    logger.info("🚀 Starting tsNeuroPredictWinMql_worker.py...")

    # Initialize OracleClient
    # OracleClient's __init__ does not accept host and port directly;
    # it retrieves these from the mql_overrides configuration internally.
    oracle_client = OracleClient()

    # Load and preprocess data
    X, y, n_steps, n_features, feature_scaler, target_scaler = load_and_preprocess_data(
        symbol=SYMBOLS[0], # Assuming single symbol for now
        timeframe_str=TIMEFRAME, # Pass the string timeframe
        num_candles=NUM_CANDLES,
        look_back=LOOK_BACK,
        prediction_horizon=PREDICTION_HORIZON,
        features_to_use=FEATURES_TO_USE,
        target_feature=TARGET_FEATURE,
        normalization_method=NORMALIZATION_METHOD
    )

    if X is None:
        logger.error("❌ Data loading and preprocessing failed. Exiting.")
        sys.exit(1)

    # Split data (worker only needs train/val for its trials)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42)
    logger.info(f"Data split: Train {len(X_train)} samples, Validation {len(X_val)} samples.")

    # Determine num_classes dynamically from y's shape
    # If y is (num_samples,), it's 1. If y is (num_samples, N), it's N.
    if y.ndim == 1:
        num_classes = 1
    else:
        num_classes = y.shape[-1] # This will be 7 if target.shape=(None, 7)
    logger.info(f"Determined num_classes for model output: {num_classes}")

    # Convert to TensorFlow Datasets or PyTorch Tensors
    if MLTUNE_BACKEND == 'tensorflow':
        # Reduced batch size to mitigate OOM errors
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
        input_shape = (n_steps, n_features)
    elif MLTUNE_BACKEND == 'pytorch':
        import torch # Import torch here
        train_dataset = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_dataset = (torch.tensor(X_val).float(), torch.tensor(y_val).float())
        input_shape = (n_steps, n_features)
    else:
        logger.error(f"Unsupported MLTUNE_BACKEND: {MLTUNE_BACKEND}")
        sys.exit(1)

    # Initialize CMdtunerSelector for the worker
    logger.info(f"Worker Initializing CMdtunerSelector with backend: {MLTUNE_BACKEND} and tuner type: {MLTUNE_TUNER_TYPE}")
    tuner_config = CMdtunerSelector(
        tuner_type=MLTUNE_TUNER_TYPE,
        backend=MLTUNE_BACKEND,
        oracle_client=oracle_client,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        project_name=MODEL_NAME, # Workers also need project_name for logging/directories
        max_trials=tune_params.get('num_trials', 1), # Workers typically run one trial at a time
        hypermodel_params=all_params # Pass all_params to the tuner for configuration
    )

    # Run the worker's tuning process (which will fetch trials from Oracle)
    logger.info("Worker starting its tuning process...")
    tuner_config.run() # Call the 'run' method for workers

    logger.info("🏁 tsNeuroPredictWinMql_worker.py finished.")


if __name__ == "__main__":
    # Ensure MetaTrader5 is initialized and finalized
    if not mt5.initialize():
        # Use a print statement here or a temporary logger since the main logger might not be fully configured yet
        print("ERROR: mt5.initialize() failed, error code =", mt5.last_error())
        sys.exit(1)
    else:
        print("INFO: MetaTrader5 initialized successfully.") # Use print for early messages

    try:
        # Run the main function, passing the logger to it
        main(logger)
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")
