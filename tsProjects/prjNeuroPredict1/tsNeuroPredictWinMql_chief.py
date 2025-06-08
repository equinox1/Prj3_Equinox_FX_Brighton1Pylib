#!/usr/bin/env python3
# +------------------------------------------------------------------+\
# |                                    tsNeuroPredictWinMql_chief.py |\
# |                                                    Tony Shepherd |\
# |                                    https://www.xercescloud.co.uk |\
# +------------------------------------------------------------------+\


import os
import sys
import logging # Import logging first
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
from tsMqlSetup import CMqlSetup # Import CMqlSetup early
from tsMqlOverrides import CMqlOverrides # Ensure CMqlOverrides is imported early
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr

from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlMLProcess import CDMLProcess # Ensure CDMLProcess is imported
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
from tsMqlMLTuner.tsMqlMLTunerMod import CMdtuner # For TensorFlow
from tsMqlMLTuner.tsMqlMLTunerModTorch import PyTorchTuner # For PyTorch


# Keras Tuner components for manual trial management
from keras_tuner.engine.trial import TrialStatus


# Import mixed_precision
from tensorflow.keras import mixed_precision

# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNER_ID"] = "chief" # This is important for the client to identify itself

# --- DETERMINE BACKEND FROM ENVIRONMENT VARIABLE FIRST ---
MLTUNE_BACKEND = os.environ.get('MLTUNE_BACKEND', 'tensorflow').lower() # Default to 'tensorflow'


# Dynamically determine num_cores and num_threads for optimal performance.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

# Initialize CMqlSetup for general configuration parameters first
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

# Set up global logging using a dedicated CMqlSetup instance
# This block configures the root logger, so it should run before any logger.getLogger(__name__) calls
clientlog_config = CMqlSetup() # Create a separate instance for logging setup

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for Chief. Using default logging.")

# Initialize logger AFTER setup_logging has been called
logger = logging.getLogger(__name__)
logger.info(f"🔧 Detected tuning backend from environment: {MLTUNE_BACKEND}")
# -- end of logging setup ----


# -- Suppress ONNX Windows version warning --
import warnings
warnings.filterwarnings("ignore", message="Unsupported Windows version")

# -- Load environment variables first --
env_trials = int(os.environ.get("MLTUNE_TRIALS", 128))

# -- Apply overrides before config extraction --
mql_overrides = CMqlOverrides()

# IMPORTANT: Override the backend using the value from the environment variable
mql_overrides.env.override_params({
    "mltune": {
        "backend": MLTUNE_BACKEND, # Use the backend detected from env
        "num_trials": env_trials,
        "tuner_type": os.environ.get('MLTUNE_TUNER_TYPE', 'hyperband'), # Get tuner type from env
        "reset_trials": True, # Ensure chief always resets trials for a fresh start
        "overwrite": True, # Ensure tuner directory is overwritten
    }
})

# Use CMqlEnvMgr to get all parameters after overrides
env_mgr = CMqlEnvMgr()
all_params = env_mgr.all_params()

# Update parameters based on overrides
app_params = all_params.get("app", {})
ml_params = all_params.get("ml", {})
tune_params = all_params.get("mltune", {}) # Re-fetch updated tune_params

logger.info(f"Using MLTUNE_BACKEND: {MLTUNE_BACKEND}")
MLTUNE_TUNER_TYPE = tune_params.get('tuner_type', 'hyperband') # Get updated tuner type
logger.info(f"Using MLTUNE_TUNER_TYPE: {MLTUNE_TUNER_TYPE}")


xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')


# --- Set mixed precision policy if using TensorFlow backend ---
if MLTUNE_BACKEND == 'tensorflow':
    try:
        mixed_precision.set_global_policy('mixed_bfloat16')
        logger.info("✅ TensorFlow mixed precision policy set to 'mixed_bfloat16'.")
    except Exception as e:
        logger.warning(f"⚠️ Could not set mixed precision policy: {e}. Falling back to default float type.")


# --- Global Configuration ---
# Extract necessary parameters for Chief
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
# MLTUNE_BACKEND is already defined at the top
# MLTUNE_TUNER_TYPE is already defined above
# ORACLE_HOST and ORACLE_PORT are already defined via app_params, but OracleClient uses internal config.
# Keeping these for clarity if they were to be used elsewhere, but not for OracleClient init.
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
        return None, None, None, None, None, None # Returns 6 Nones. This path is fine.

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
        return None, None, None, None, None, None # Returns 6 Nones. This path is fine.

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
        return None, None, None, None, None, None # Returns 6 Nones. This path is fine.

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

    # --- Debugging: Add checks for X and y from Create_Xy_input_and_target ---
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        logger.error(f"❌ CDMLProcess.Create_Xy_input_and_target did not return numpy arrays. X type: {type(X)}, y type: {type(y)}")
        return None, None, None, None, None, None # Return 6 Nones if types are wrong

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

    logger.info("✅ Data preprocessing complete. Preparing return values.")
    # Debugging: Log the values about to be returned
    # Explicitly verify each component right before the return
    # The error is not here, but in how the main function unpacks this return.
    returned_values = (X_scaled, y_scaled, n_steps, n_features, feature_scaler, target_scaler)
    logger.debug(f"Returning: {len(returned_values)} values. Types: {[type(val) for val in returned_values]}")
    logger.debug(f"X_scaled shape={X_scaled.shape}, y_scaled shape={y_scaled.shape}, n_steps={n_steps}, n_features={n_features}, feature_scaler_type={type(feature_scaler)}, target_scaler_type={type(target_scaler)}")
    
    return returned_values


def main(logger):
    logger.info("🚀 Starting tsNeuroPredictWinMql_chief.py...")

    # Initialize OracleClient
    # OracleClient's __init__ does not accept host and port directly;
    # it retrieves these from the mql_overrides configuration internally.
    oracle_client = OracleClient()

    # Load and preprocess data
    X, y, n_steps, n_features, feature_scaler, target_scaler = load_and_preprocess_data(
        symbol=SYMBOLS[0], # Assuming single symbol for now
        timeframe_str=TIMEFRAME,
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

    # Corrected Data Splitting
    # First, split into training set and a combined validation/test set
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Initial data split: Train {len(X_train)} samples, Validation/Test {len(X_val_test)} samples.")

    # Second, split the combined validation/test set into separate validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42 # 0.5 of the 20% = 10% for test, 10% for val
    )
    logger.info(f"Final data split: Train {len(X_train)} samples, Validation {len(X_val)} samples, Test {len(X_test)} samples.")


    # Determine num_classes dynamically from y's shape
    if y.ndim == 1:
        num_classes = 1
    else:
        num_classes = y.shape[-1]
    logger.info(f"Determined num_classes for model output: {num_classes}")


    # Convert to TensorFlow Datasets or PyTorch Tensors
    if MLTUNE_BACKEND == 'tensorflow':
        # Reduced batch size to mitigate OOM errors
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)
        input_shape = (n_steps, n_features)
    elif MLTUNE_BACKEND == 'pytorch':
        import torch # Import torch here
        train_dataset = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_dataset = (torch.tensor(X_val).float(), torch.tensor(y_val).float())
        test_dataset = (torch.tensor(X_test).float(), torch.tensor(y_test).float())
        input_shape = (n_steps, n_features)
    else:
        logger.error(f"Unsupported MLTUNE_BACKEND: {MLTUNE_BACKEND}")
        sys.exit(1)

    # Initialize CMdtunerSelector
    logger.info(f"Chief Initializing CMdtunerSelector with backend: {MLTUNE_BACKEND} and tuner type: {MLTUNE_TUNER_TYPE}")
    tuner_config = CMdtunerSelector(
        tuner_type=MLTUNE_TUNER_TYPE,
        backend=MLTUNE_BACKEND, # Use the correctly detected MLTUNE_BACKEND
        oracle_client=oracle_client,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset, # Pass test_dataset for final evaluation
        input_shape=input_shape,
        num_classes=num_classes,
        project_name=MODEL_NAME,
        max_trials=tune_params.get('num_trials', 64),
        overwrite=tune_params.get('overwrite', True),
        hypermodel_params=all_params # Pass all_params to the tuner for configuration
    )

    logger.info("Chief starting tuning process...")
    best_model = tuner_config.run()

    if best_model:
        logger.info("✅ Best model found and exported by Chief.")
        
        # Evaluate the best model on the test dataset
        logger.info("📈 Evaluating the best model on the test dataset...")
        test_loss, test_metrics = tuner_config.evaluate_best_model(best_model, test_dataset)
        
        # Check if test_metrics is a dictionary before iterating
        if test_loss is not None and isinstance(test_metrics, dict):
            logger.info(f"Final Test Loss: {test_loss}")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"Final Test {metric_name}: {metric_value}")
        else:
            logger.warning("⚠️ Failed to retrieve valid test evaluation results or test_metrics is not a dictionary. Skipping metrics display.")

        # Optional: ONNX conversion for TensorFlow models
        if MLTUNE_BACKEND == 'tensorflow':
            try:
                import tf2onnx # Import here for lazy loading
                import onnx
                from onnx import checker
                import onnxruntime as ort

                # Define ONNX export path
                base_params = all_params.get("base", {})
                global_logdir = base_params.get('mp_glob_base_log_path', Path(__file__).parent.parent.parent / 'Logdir')
                
                onnx_path = Path(global_logdir) / xerces_servername / MLTUNE_BACKEND / f"{MODEL_NAME}.onnx"
                onnx_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Attempting to convert TensorFlow model to ONNX at {onnx_path}")
                # Ensure input_signature is correct for your model
                # Assuming `input_shape` is (timesteps, features)
                # Keras models expect a batch dimension as the first dimension
                input_signature = [
                    tf.TensorSpec(shape=(None, input_shape[0], input_shape[1]), dtype=tf.float32, name="input")
                ]
                
                # Convert the Keras model (best_model is a tf.keras.Model)
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=13)
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                logger.info("✅ TensorFlow model successfully converted to ONNX.")

                # Check ONNX model
                onnx_model_checked = onnx.load(onnx_path)
                checker.check_model(onnx_model_checked)
                logger.info("✅ ONNX model check successful.")

                # Optional: Run inference with ONNX Runtime to verify
                ort_session = ort.InferenceSession(onnx_path)
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
        # Use print for early messages before main logger is fully configured
        print("ERROR: mt5.initialize() failed, error code =", mt5.last_error())
        sys.exit(1)
    else:
        print("INFO: MetaTrader5 initialized successfully.")

    try:
        # Run the main function, passing the logger to it
        main(logger)
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")
