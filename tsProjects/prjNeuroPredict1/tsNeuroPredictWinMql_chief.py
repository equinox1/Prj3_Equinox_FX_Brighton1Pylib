#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                    tsNeuroPredictWinMql_chief.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
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
import tf2onnx
import onnx
from onnx import checker
import onnxruntime as ort
import MetaTrader5 as mt5

# Custom modules

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
from tsMqlMLTuner.tsMqlMLTunerModTorch import PyTorchTuner

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for Chief. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
# -- end of logging setup ----



from tsMqlSetup import CMqlSetup
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



from tsMqlSetup import CMqlSetup
gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
logger.info(f"Chief get: Using backend: {backend} and tuner type: {gtuner_model}")
print(f"DEBUG: Using backend: {backend} and tuner type: {gtuner_model}")
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

logger.info(f"Chief Using backend: {backend}")
logger.info(f"Chief Using tuner model: {gtuner_model}")

# -- end of logging setup ----


# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNER_ID"] = "chief" # This is important for the client to identify itself

# --- Global Configuration ---
# Use CMqlEnvMgr to get all parameters
env_mgr = CMqlEnvMgr()
all_params = env_mgr.all_params()
base_params = all_params.get("base", {})
#app_params = all_params.get("app", {})
broker_params = all_params.get("broker", {})
ml_params = all_params.get("ml", {})
#tune_params = all_params.get("mltune", {})

# Extract necessary parameters for Chief
SYMBOLS = app_params.get('mp_app_symbols', ['EURUSD'])
TIMEFRAME = app_params.get('mp_app_timeframe', 'M1')
NUM_CANDLES = app_params.get('mp_app_num_candles', 10000)
MODEL_NAME = app_params.get('mp_glob_sub_ml_model_name', 'ts_mql_model')
LOOK_BACK = ml_params.get('mp_ml_look_back', 60)
PREDICTION_HORIZON = ml_params.get('mp_ml_prediction_horizon', 1)
TRAIN_SPLIT_RATIO = ml_params.get('mp_ml_train_split_ratio', 0.8)
# CORRECTED: Updated FEATURES_TO_USE to use prefixed column names
FEATURES_TO_USE = ml_params.get('mp_ml_features_to_use', ['R1_Open', 'R1_High', 'R1_Low', 'R1_Close', 'R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume'])
# CORRECTED: Updated TARGET_FEATURE to use prefixed column name
TARGET_FEATURE = ml_params.get('mp_ml_target_feature', 'R1_Close')
NORMALIZATION_METHOD = ml_params.get('mp_ml_normalization_method', 'StandardScaler')
MODEL_TYPE = ml_params.get('mp_ml_model_type', 'LSTM') # Default to LSTM
MLTUNE_BACKEND = backend
logger.info(f"Using MLTUNE_BACKEND: {MLTUNE_BACKEND}")
MLTUNE_NUM_TRIALS = tune_params.get('num_trials', 10)
MLTUNE_TUNER_TYPE = gtuner_model
logger.info(f"Using MLTUNE_TUNER_TYPE: {MLTUNE_TUNER_TYPE}")

MLTUNE_OVERWRITE = tune_params.get('overwrite', True)
MLTUNE_RESET_TRIALS = tune_params.get('reset_trials', True)
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
    # You might need to adjust this based on your data loading strategy (ticks vs rates, API vs file)
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

    # Check if processed_data_df is empty after processing
    if processed_data_df.empty:
        logger.error("❌ Data processing resulted in an empty DataFrame.")
        return None, None, None, None, None, None

    # DEBUGGING: Print columns of the DataFrame after CDataProcess
    logger.info(f"DEBUG: Columns after CDataProcess: {processed_data_df.columns.tolist()}")

    logger.info("⚙️ Creating ML sequences (CDMLProcess)...")
    # Initialize CDMLProcess for creating sequences
    # CDMLProcess also takes look_back, prediction_horizon, features_to_use, target_feature
    # It's good practice to pass these explicitly if CDMLProcess uses them for sequence creation
    ml_process = CDMLProcess(
        look_back=look_back,
        prediction_horizon=prediction_horizon,
        features_to_use=features_to_use,
        target_feature=target_feature
    )

    # Now, create sequences using the ml_process instance
    # CORRECTED: Using Create_Xy_input_and_target instead of create_sequences
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

    # Determine n_steps and n_features from the created X array
    # X will have shape (num_samples, back_window, num_features)
    n_steps = X.shape[1]
    n_features = X.shape[2]
    logger.info(f"✅ Sequences created. X shape: {X.shape}, y shape: {y.shape}")

    # Reshape y for scaling if it's a 1D array
    if y.ndim == 1:
        y_reshaped_for_scaler = y.reshape(-1, 1)
    else:
        y_reshaped_for_scaler = y

    # Initialize scaler for features (X)
    feature_scaler = StandardScaler()
    # Reshape X to 2D for scaling, then back to 3D
    X_scaled = feature_scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
    logger.info(f"X scaled shape: {X_scaled.shape}")

    # Initialize scaler for target (y)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_reshaped_for_scaler)

    # If y was 1D, convert it back to 1D after scaling
    if y.ndim == 1:
        y_scaled = y_scaled.flatten()
    logger.info(f"y scaled shape: {y_scaled.shape}")

    logger.info("✅ Data preprocessing complete.")
    return X_scaled, y_scaled, n_steps, n_features, feature_scaler, target_scaler


def main(logger):
    logger.info("🚀 Starting tsNeuroPredictWinMql_chief.py...")

    # Initialize OracleClient
    oracle_client = OracleClient(host=ORACLE_HOST, port=ORACLE_PORT)

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

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42)
    logger.info(f"Data split: Train {len(X_train)} samples, Validation {len(X_val)} samples.")

    # Determine num_classes dynamically from y's shape
    # If y is (num_samples,), it's 1. If y is (num_samples, N), it's N.
    if y.ndim == 1:
        num_classes = 1
    else:
        num_classes = y.shape[-1] # This will be 7 if target.shape=(None, 7)
    logger.info(f"Determined num_classes for model output: {num_classes}")

    # Convert to TensorFlow Datasets for KerasTuner (if using TensorFlow backend)
    # Or to PyTorch Tensors and DataLoader (if using PyTorch backend)
    if MLTUNE_BACKEND == 'tensorflow':
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
        input_shape = (n_steps, n_features)
    elif MLTUNE_BACKEND == 'pytorch':
        import torch # Import torch here if not already imported globally
        train_dataset = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_dataset = (torch.tensor(X_val).float(), torch.tensor(y_val).float())
        input_shape = (n_steps, n_features)
    else:
        logger.error(f"Unsupported MLTUNE_BACKEND: {MLTUNE_BACKEND}")
        sys.exit(1)


    # Initialize CMdtunerSelector
    logger.info(f"Chief Initializing CMdtunerSelector with backend: {MLTUNE_BACKEND} and tuner type: {MLTUNE_TUNER_TYPE}")
    tuner_config = CMdtunerSelector(
        tuner_type=MLTUNE_TUNER_TYPE,
        backend=MLTUNE_BACKEND,
        oracle_client=oracle_client, # Pass the oracle_client instance
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        project_name=MODEL_NAME,
        max_trials=MLTUNE_NUM_TRIALS,
        hypermodel_params=all_params # Pass all_params here
    )

    # --- DEBUGGING: Print type of tuner_config before calling run_search ---
    logger.info(f"Type of tuner_config before run_search: {type(tuner_config)}")
    print(f"DEBUG: Type of tuner_config before run_search: {type(tuner_config)}")
    # --- END DEBUGGING ---

    # Run the tuner search
    logger.info("Starting hyperparameter search...")
    best_model = tuner_config.run_search()

    if best_model:
        logger.info("✅ Hyperparameter search completed. Best model obtained.")

        # --- Final Evaluation and Prediction ---
        logger.info("Evaluating the best model...")
        try:
            # Prepare the full dataset for final prediction/evaluation
            # Ensure X_val and y_val are in the correct format for prediction
            if MLTUNE_BACKEND == 'tensorflow':
                # For TensorFlow, predict directly on the numpy array X_val
                predictions_scaled = best_model.predict(X_val)
            elif MLTUNE_BACKEND == 'pytorch':
                # For PyTorch, convert X_val to tensor and move to appropriate device
                import torch # Ensure torch is imported
                best_model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val).float()
                    predictions_scaled = best_model(X_val_tensor).numpy() # Convert back to numpy
            else:
                logger.error(f"Unsupported backend for prediction: {MLTUNE_BACKEND}")
                predictions_scaled = None

            if predictions_scaled is not None:
                # Inverse transform predictions and actual values
                # Ensure predictions_scaled has the correct shape for inverse_transform
                if predictions_scaled.ndim == 1:
                    predictions_scaled_reshaped = predictions_scaled.reshape(-1, 1)
                else:
                    predictions_scaled_reshaped = predictions_scaled

                if y_val.ndim == 1:
                    y_val_reshaped = y_val.reshape(-1, 1)
                else:
                    y_val_reshaped = y_val

                predictions = target_scaler.inverse_transform(predictions_scaled_reshaped)
                actuals = target_scaler.inverse_transform(y_val_reshaped)

                # Flatten if they were originally 1D
                if y.ndim == 1:
                    predictions = predictions.flatten()
                    actuals = actuals.flatten()

                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                r2 = r2_score(actuals, predictions)

                logger.info(f"Final Model Evaluation:")
                logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
                logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
                logger.info(f"  R-squared (R2): {r2:.4f}")

                # Plotting predictions vs actuals
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=range(len(actuals)), y=actuals, label='Ground Truth')
                sns.lineplot(x=range(len(predictions)), y=predictions, label='Predictions')
                plt.title("Prediction vs Ground Truth")
                plt.xlabel("Time Step")
                plt.ylabel(TARGET_FEATURE)
                plt.legend()
                plt.grid(True)
                modeldatapath = base_params.get('mp_glob_sub_ml_src_modeldata')
                modelname = base_params.get('mp_glob_sub_ml_model_name')
                plot_path = os.path.join(modeldatapath, f"{modelname}_predictions.png")
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"📊 Prediction plot saved: {plot_path}")

        except Exception as e:
            logger.error(f"❌ Error during final prediction/evaluation: {e}")

        # Save model
        modeldatapath = base_params.get('mp_glob_sub_ml_src_modeldata')
        modelname = base_params.get('mp_glob_sub_ml_model_name')
        # Ensure modelname is valid for filename
        if not modelname:
            modelname = "default_model"
            logger.warning("Model name not found in config, using 'default_model'.")

        model_save_path = os.path.join(modeldatapath, f"{modelname}.h5")

        if MLTUNE_BACKEND == 'tensorflow':
            try:
                best_model.save(model_save_path)
                logger.info(f"✅ TensorFlow model saved: {model_save_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save TensorFlow model: {e}")
        elif MLTUNE_BACKEND == 'pytorch':
            try:
                import torch # Ensure torch is imported
                # For PyTorch, save the state_dict
                torch.save(best_model.state_dict(), model_save_path.replace('.h5', '.pth'))
                logger.info(f"✅ PyTorch model state_dict saved: {model_save_path.replace('.h5', '.pth')}")
            except Exception as e:
                logger.error(f"❌ Failed to save PyTorch model state_dict: {e}")
        else:
            logger.warning(f"Model saving not implemented for backend: {MLTUNE_BACKEND}")


        # Optional: Convert to ONNX (TensorFlow only for now)
        if MLTUNE_BACKEND == 'tensorflow' and app_params.get("mp_app_ONNX_save", False):
            try:
                import tf2onnx
                import onnx
                from onnx import checker
                logger.info("Attempting to convert TensorFlow model to ONNX...")
                # Define input signature for ONNX conversion
                spec = [tf.TensorSpec(best_model.input_shape, tf.float32, name="input")]
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=13)
                onnx_path = os.path.join(modeldatapath, f"{modelname}.onnx")
                onnx.save(onnx_model, onnx_path)
                logger.info(f"✅ ONNX model saved: {onnx_path}")

                # Check ONNX model
                onnx_model_checked = onnx.load(onnx_path)
                checker.check_model(onnx_model_checked)
                logger.info("✅ ONNX model check successful.")

                # Optional: Run inference with ONNX Runtime to verify
                ort_session = ort.InferenceSession(onnx_path)
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name

                # Use a small subset of X_val for ONNX inference test
                test_input = X_val[:1].astype(np.float32)
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
        logger.error("❌ mt5.initialize() failed, error code =", mt5.last_error())
        sys.exit(1)
    else:
        logger.info("✅ MetaTrader5 initialized successfully.")

    try:
        # Run the main function
        main(logger)
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")
