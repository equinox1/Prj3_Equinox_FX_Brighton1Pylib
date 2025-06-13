#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                            tsNeuroPredictWinMql_worker.py        |
# |                        Refactored with CMdtunerSelector          |
# +------------------------------------------------------------------+

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging


# --- Global Configuration & Logger Setup ---
# Initialize CMqlSetup for the worker itself
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=_logical_cores # Use logical cores for threads
)
setup_config.setup_logging() # Configure logging for this script

logger = logging.getLogger(__name__)

# Load environment variables and app parameters using CMqlOverrides early
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

# Worker-specific configuration based on passed arguments or defaults
# tuner_id is typically passed as a command-line argument to the worker process
import sys
if len(sys.argv) > 1:
    tuner_id = sys.argv[1]
else:
    tuner_id = "worker_default"
    logger.warning("No tuner_id provided as command-line argument. Using default.")

SYMBOL = app_params.get('SYMBOL', 'EURUSD')
TIMEFRAME = getattr(mt5, app_params.get('TIMEFRAME', 'TIMEFRAME_H1'))
START_DATE_STR = app_params.get('START_DATE', '2023-01-01')
END_DATE_STR = app_params.get('END_DATE', '2023-12-31')
MODEL_NAME = app_params.get('MODEL_NAME', 'tsneuromodel')
DATA_PATH = app_params.get('DATA_PATH', 'tsModelData')
TRAIN_SPLIT = app_params.get('TRAIN_SPLIT', 0.8)
VAL_SPLIT = app_params.get('VAL_SPLIT', 0.1)
PRED_HORIZON = app_params.get('PRED_HORIZON', 1)
MLTUNE_BACKEND = tune_params.get('backend', 'tensorflow').lower() # tensorflow or pytorch
EPOCHS = tune_params.get('epochs', 10)
BATCH_SIZE = tune_params.get('batch_size', 32)
NUM_TRIALS_WORKER = 1 # Workers typically run one trial at a time
OVERWRITE = tune_params.get('overwrite', False)


logger.info(f"🔧 Worker {tuner_id} backend set to: {MLTUNE_BACKEND}")
logger.info(f"🔧 Project directory: {Path(DATA_PATH) / MODEL_NAME}")


# Set mixed precision policy based on configuration
policy_name = setup_config.precision
if policy_name == 'mixed_float16':
    policy = mixed_precision.Policy('mixed_float16')
elif policy_name == 'mixed_bfloat16':
    policy = mixed_precision.Policy('mixed_bfloat16')
else:
    policy = mixed_precision.Policy('float32') # Default to float32
mixed_precision.set_global_policy(policy)
logger.info(f"✨ Global mixed precision policy set to: {mixed_precision.global_policy().name}")


def create_sequences(data, seq_length, pred_horizon):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_horizon)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    logger.info(f"🚀 Starting tsNeuroPredictWinMql_worker.py with tuner_id: {tuner_id}")

    # ----------------------------
    # Data Loading and Preprocessing
    # ----------------------------
    logger.info("📊 Loading and preprocessing data...")
    data_loader = CDataLoader(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR, DATA_PATH)
    data_df = data_loader.load_data()

    if data_df is None or data_df.empty:
        logger.error("❌ Failed to load data or data is empty.")
        sys.exit(1)

    data_process = CDataProcess(data_df)
    processed_data = data_process.process_data()

    if processed_data is None or processed_data.empty:
        logger.error("❌ Processed data is empty.")
        sys.exit(1)

    # Convert DataFrame to NumPy array for scaling and sequence creation
    features = processed_data[['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].values
    labels = processed_data[['open', 'high', 'low', 'close']].values # Example: predict next 4 price points

    # Scale features and labels
    feature_scaler = StandardScaler()
    labels_scaler = StandardScaler()

    scaled_features = feature_scaler.fit_transform(features)
    scaled_labels = labels_scaler.fit_transform(labels) # Scale labels as well

    # Determine sequence length based on some configuration (e.g., from tune_params)
    SEQ_LENGTH = tune_params.get('sequence_length', 60) # Default to 60 if not specified

    # Create sequences
    X, y = create_sequences(scaled_features, SEQ_LENGTH, PRED_HORIZON)

    # Reshape y to be 3D (samples, pred_horizon, num_label_features) for consistency
    num_label_features = scaled_labels.shape[1]
    y_reshaped = []
    for i in range(len(scaled_labels) - SEQ_LENGTH - PRED_HORIZON + 1):
        y_reshaped.append(scaled_labels[(i + SEQ_LENGTH):(i + SEQ_LENGTH + PRED_HORIZON)])
    y_reshaped = np.array(y_reshaped)

    if X.shape[0] == 0 or y_reshaped.shape[0] == 0:
        logger.error("❌ Not enough data to create sequences. Adjust SEQ_LENGTH or data range.")
        sys.exit(1)

    # Determine num_classes (number of features in the output sequence for prediction)
    num_classes = y_reshaped.shape[-1]
    logger.info(f"Dynamically determined num_classes (output features): {num_classes}")

    # Split data into training, validation, and test sets
    # Workers only need train and val for tuning, but test_dataset is passed for consistency
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_reshaped, test_size=1 - TRAIN_SPLIT - VAL_SPLIT, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_SPLIT/(TRAIN_SPLIT + VAL_SPLIT), random_state=42)

    logger.info(f"Dataset shapes: X_train:{X_train.shape}, y_train:{y_train.shape}")
    logger.info(f"Dataset shapes: X_val:{X_val.shape}, y_val:{y_val.shape}")
    logger.info(f"Dataset shapes: X_test:{X_test.shape}, y_test:{y_test.shape}")

    # Ensure datasets are TensorFlow tf.data.Dataset objects for CMdtunerSelector
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)


    # Determine input_shape for the model
    input_shape = X_train.shape[1:]
    logger.info(f"Dynamically determined input_shape for model: {input_shape}")

    # ----------------------------
    # OracleClient and Tuner Setup
    # ----------------------------
    logger.info("Setting up Oracle Client...")
    oracle_client = OracleClient(
        host=app_params.get('xerces_server', '127.0.0.1'),
        port=app_params.get('xerces_port', 9000)
    )

    # Initialize CMdtunerSelector
    logger.info(f"Initializing CMdtunerSelector (Worker {tuner_id})...")
    tuner_config = CMdtunerSelector(
        tuner_id=tuner_id,
        backend=MLTUNE_BACKEND, # Pass the correctly detected MLTUNE_BACKEND
        oracle_client=oracle_client,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset, # Pass test_dataset for final evaluation
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        project_name=MODEL_NAME, # Workers also need project_name for logging/directories
        max_trials=NUM_TRIALS_WORKER, # Workers typically run one trial at a time
        overwrite=OVERWRITE, # Pass the 'overwrite' argument
        hypermodel_params=all_params # Pass all_params to the tuner for configuration
        # IMPORTANT: Do NOT pass 'tuner_type' as a direct keyword argument here.
        # It is already part of 'hypermodel_params' within the 'mltune' key.
    )

    # Run the worker's tuning process (which will fetch trials from Oracle)
    logger.info(f"Worker {tuner_id} starting its tuning process...")
    tuner_config.run() # Call the 'run' method for workers

    logger.info(f"🏁 tsNeuroPredictWinMql_worker.py finished for tuner_id: {tuner_id}.")


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
