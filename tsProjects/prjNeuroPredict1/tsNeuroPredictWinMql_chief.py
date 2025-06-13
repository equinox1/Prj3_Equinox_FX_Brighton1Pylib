#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                    tsNeuroPredictWinMql_chief.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+

import os
import sys
import logging # Import logging, but do NOT configure the root logger here.
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
import tf2onnx
import onnx
from onnx import checker
import onnxruntime as ort
import MetaTrader5 as mt5

# Custom modules
from tsMqlSetup import CMqlSetup # Import CMqlSetup for non-logging config, but not for root logger setup.
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr

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
# Initialize CMqlSetup for the chief itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
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

# Use the loaded parameters
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
NUM_TRIALS = tune_params.get('num_trials', 1)
OVERWRITE = tune_params.get('overwrite', False)


logger.info(f"🔧 MLTUNE_BACKEND set to: {MLTUNE_BACKEND}")
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
    logger.info("🚀 Starting tsNeuroPredictWinMql_chief.py")

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
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_reshaped, test_size=1 - TRAIN_SPLIT - VAL_SPLIT, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_SPLIT/(TRAIN_SPLIT + VAL_SPLIT), random_state=42)

    logger.info(f"Dataset shapes: X_train:{X_train.shape}, y_train:{y_train.shape}")
    logger.info(f"Dataset shapes: X_val:{X_val.shape}, y_val:{y_val.shape}")
    logger.info(f"Dataset shapes: X_test:{X_test.shape}, y_test:{y_test.shape}")

    # Ensure datasets are TensorFlow tf.data.Dataset objects for CMdtunerSelector
    # For PyTorch, these will be converted to DataLoader internally by PyTorchTuner
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
    logger.info("Initializing CMdtunerSelector (Chief)...")
    tuner_config = CMdtunerSelector(
        tuner_id="chief",
        backend=MLTUNE_BACKEND,
        oracle_client=oracle_client,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset, # Pass test_dataset for final evaluation
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        project_name=MODEL_NAME, # Workers also need project_name for logging/directories
        max_trials=NUM_TRIALS, # Chief should typically run all trials
        overwrite=OVERWRITE, # Pass the 'overwrite' argument
        hypermodel_params=all_params # Pass all_params to the tuner for configuration
        # IMPORTANT: Do NOT pass 'tuner_type' as a direct keyword argument here.
        # It is already part of 'hypermodel_params' within the 'mltune' key.
    )

    # Run the tuning process
    logger.info("Chief starting its tuning process...")
    best_model = tuner_config.run() # Call the 'run' method for the chief

    if best_model:
        logger.info("✅ Best model found and exported by Chief.")
        # ----------------------------
        # Final Evaluation on Test Set
        # ----------------------------
        logger.info("📈 Evaluating the best model on the test dataset...")
        test_loss, test_metrics = tuner_config.evaluate_best_model(best_model, (X_test, y_test) if MLTUNE_BACKEND == 'pytorch' else test_dataset)

        if test_metrics and isinstance(test_metrics, dict):
            logger.info("📊 Test Evaluation Results:")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.5f}")

            # Optionally, save a report to a file
            report_path = Path(DATA_PATH) / MODEL_NAME / "test_evaluation_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(test_metrics, f, indent=4)
            logger.info(f"📋 Test evaluation report saved to {report_path}")
        else:
            logger.warning("⚠️ Failed to retrieve valid test evaluation results or test_metrics is not a dictionary. Skipping metrics display.")

        # ----------------------------
        # Model Saving and Conversion (for TF only)
        # ----------------------------
        if MLTUNE_BACKEND == 'tensorflow':
            logger.info("💾 Saving and converting TensorFlow model...")
            model_save_path = Path(DATA_PATH) / MODEL_NAME / "best_model_tf"
            model_save_path.mkdir(parents=True, exist_ok=True)
            best_model.save(model_save_path)
            logger.info(f"✅ TensorFlow model saved to {model_save_path}")

            # Attempt ONNX conversion
            try:
                logger.info("🔄 Attempting to convert TensorFlow model to ONNX...")
                onnx_model_path = Path(DATA_PATH) / MODEL_NAME / "best_model.onnx"
                spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, spec, opset=13, output_path=str(onnx_model_path))
                logger.info(f"✅ ONNX model converted and saved to {onnx_model_path}")

                # Verify ONNX model
                logger.info("🔍 Verifying ONNX model...")
                onnx.checker.check_model(onnx_model)
                logger.info("✅ ONNX model verification successful.")

                # Test ONNX Runtime inference
                logger.info("🧪 Testing ONNX Runtime inference...")
                ort_session = ort.InferenceSession(str(onnx_model_path))
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name

                # Use a small subset of X_val for ONNX inference test
                # Ensure the test_input has the correct batch dimension (None, timesteps, features)
                test_input = X_val[:1].astype(np.float32) # Get first sample, ensure float32
                if test_input.ndim == 2: # If input is (timesteps, features), add batch dim
                    test_input = np.expand_dims(test_input, axis=0)

                ort_outs = ort_session.run([output_name], {input_name: test_input})
                logger.info(f"✅ ONNX Runtime inference test successful. Output shape: {ort_outs[0].shape}")

            except ImportError:
                logger.warning("tf2onnx or onnx not installed. Skipping ONNX conversion.")
            except Exception as e:
                logger.error(f"❌ Failed to convert or verify ONNX model: {e}")
    else:
        logger.info("Skipping final evaluation and model saving as no best model was found.")

    logger.info("🏁 tsNeuroPredictWinMql_chief.py finished.")


if __name__ == "__main__":
    # Ensure MetaTrader5 is initialized and finalized
    if not mt5.initialize():
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
