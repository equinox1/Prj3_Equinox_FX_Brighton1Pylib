#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                    tsNeuroPredictWinMql_chief.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+\
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

# Import MetaTrader5 at the very top to ensure it's available globally
import MetaTrader5 as mt5

# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

# Import torch and related modules for PyTorch backend
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
# Custom modules
from tsMqlSetup import CMqlSetup # Import CMqlSetup for non-logging config, but not for root logger setup.
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr

from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess # Assuming this file exists and contains CDataProcess
from tsMqlMLProcess import CDMLProcess

# Distributed tuner system
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector


# Keras Tuner components for manual trial management
from keras_tuner.engine.trial import TrialStatus



# Load environment variables and app parameters early
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get('base', {}) # Get base_params

params_dict = {
    "hypermodel_params": all_params, # Pass all_params so app_params and tune_params are available
    "dataset_params": app_params.get('mp_app_dataset_params', {}),
    "base_path": base_params.get('mp_glob_base_log_path'), # Use base_params for base_path
    "model_id": app_params.get('mp_app_model_id', 'tsneuromodel_1')
}

# Ensure logging is configured using tsMqlLogService.CMLogServiceSetup
from tsMqlLogService import CMLogServiceSetup
# Get the backend from environment, tune_params, or default to pytorch
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))
# --- NEW DEBUG LOGS FOR BACKEND ---
logger.info(f"[tsNeuroPredictWinMql_chief] BACKEND env var (raw): '{os.environ.get('BACKEND')}'")
logger.info(f"[tsNeuroPredictWinMql_chief] backend_for_log (derived for logging setup): '{backend_for_log}'")
# --- END NEW DEBUG LOGS ---

# Use the correct log directory from base_params
logdir_arg = base_params.get('mp_glob_base_log_path')
servername_arg = app_params.get('mp_app_servername', socket.gethostname())

CMLogServiceSetup.initialize_logging(
    loglevel=app_params.get('mp_app_log_level', 'INFO'),
    logdir=logdir_arg, # Use the correct log directory
    logfile=app_params.get('mp_app_log_filename', 'tsneuropredict_app.log'),
    servername=servername_arg,
    backend=backend_for_log, # Pass backend for log path
    enable_logging=app_params.get('mp_app_enable_logging', True)
)

# Apply mixed precision policy for TensorFlow if enabled
if app_params.get('mp_app_enable_mixed_precision_tf', False) and backend_for_log == 'tensorflow':
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    logger.info("TensorFlow mixed precision policy set to 'mixed_float16'.")

# Initialize utilities and reference (if needed globally)
utilities = CUtilities()
mql_ref = CMqlRefConfig()

def fetch_data(symbol, timeframe, start_date, end_date):
    if not mt5.initialize():
        logger.error("initialize() failed, error code = %s", mt5.last_error())
        mt5.shutdown()
        return None

    logger.info(f"Fetching API rates from MT5 for {symbol} from {start_date} to {end_date}...")
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.warning(f"No API rates found for {symbol} from {start_date} to {end_date}. Attempting to load local file rates...")
        try:
            from tsMqlDataLoader import CDataLoader
            from tsMqlDataProcess import CDataProcess

            mql_env = CMqlEnvMgr()
            all_params = mql_env.all_params()
            base_params = all_params.get("base", {})

            file_data_loader = CDataLoader(
                symbol=symbol,
                timeframe=timeframe,
                start_date_str=start_date.strftime('%Y-%m-%d'),
                end_date_str=end_date.strftime('%Y-%m-%d'),
                data_path=base_params.get("mp_glob_base_data_path", "Mql5Data"),
                mp_data_loadapiticks=False,
                mp_data_loadapirates=False,
                mp_data_loadfileticks=False,
                mp_data_loadfilerates=True
            )
            dfs = file_data_loader.run_dataloader_services()
            df_fallback = dfs.get('df_file_rates')

            if df_fallback is not None and not df_fallback.empty:
                logger.info(f"✅ Loaded fallback local file data for {symbol}. Shape: {df_fallback.shape}")
                return df_fallback
            else:
                logger.error("❌ Fallback local file data is also empty.")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading fallback local file data: {e}", exc_info=True)
            return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def preprocess_data(df, input_sequence_length, output_sequence_length, target_column=None):
    logger.info(f"Preprocessing DataFrame columns: {df.columns.tolist()}")
    df_cols_lower = [col.lower() for col in df.columns]

    if set(['open', 'high', 'low', 'close']).issubset(df_cols_lower):
        colmap = {col.lower(): col for col in df.columns}
        features = [colmap['open'], colmap['high'], colmap['low'], colmap['close']]
        target_column = target_column or colmap['close']
    elif set(['r2_open', 'r2_high', 'r2_low', 'r2_close']).issubset(df_cols_lower):
        features = ['R2_Open', 'R2_High', 'R2_Low', 'R2_Close']
        target_column = target_column or 'R2_Close'
    elif set(['r1_open', 'r1_high', 'r1_low', 'r1_close']).issubset(df_cols_lower):
        features = ['R1_Open', 'R1_High', 'R1_Low', 'R1_Close']
        target_column = target_column or 'R1_Close'
    else:
        raise KeyError("Could not determine valid price columns from DataFrame: " + str(df.columns.tolist()))

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaled_features = scaler_X.fit_transform(df[features])
    target_data = df[target_column].values.reshape(-1, 1)
    scaled_target = scaler_y.fit_transform(target_data)

    X, y = [], []
    for i in range(len(scaled_features) - input_sequence_length - output_sequence_length + 1):
        X.append(scaled_features[i:(i + input_sequence_length)])
        y.append(scaled_target[(i + input_sequence_length):(i + input_sequence_length + output_sequence_length)].flatten())

    return np.array(X), np.array(y), scaler_X, scaler_y


def main():
    logger.info("🚀 Starting tsNeuroPredictWinMql_chief.py...")

    # Load parameters
    symbol = app_params.get('mp_app_symbol', 'EURUSD')
    timeframe = mql_ref.mt5_timeframe_from_string(app_params.get('mp_app_timeframe', 'M1'))
    start_date_str = app_params.get('mp_app_start_date', '2023-01-01 00:00:00')
    end_date_str = app_params.get('mp_app_end_date', datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'))
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)

    input_sequence_length = app_params.get('mp_app_input_sequence_length', 60)
    output_sequence_length = app_params.get('mp_app_output_sequence_length', 1) # Predicting next 1 minute/candle

    logger.info(f"Fetching data for {symbol} ({timeframe}) from {start_date} to {end_date}...")
    df = fetch_data(symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        logger.error("Failed to fetch data or data is empty. Exiting.")
        sys.exit(1)
    logger.info(f"✅ Data fetched. Shape: {df.shape}")

    logger.info("Preprocessing data...")
    data_X, data_y, scaler_X, scaler_y = preprocess_data(df.copy(), input_sequence_length, output_sequence_length)
    logger.info(f"✅ Data preprocessed. X shape: {data_X.shape}, y shape: {data_y.shape}")

    # Split data into training and testing sets
    # Using 80/20 split, shuffle=False for time series
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, shuffle=False, random_state=42)
    logger.info(f"Data split: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")

    # Determine the backend for CMdtunerSelector from the already determined backend_for_log
    # This ensures consistency with the logging backend.
    current_backend = backend_for_log 

    # Torch conversion with correct shapes, only if backend is pytorch
    train_dataset = None
    val_dataset = None
    if current_backend == "pytorch":
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_test.squeeze(), dtype=torch.float32)

        train_dataset = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        val_dataset = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    # Use the correct oracle URL from app_params or base_params
    oracle_host = app_params.get('xerces_server', '192.168.1.103')
    oracle_port = app_params.get('xerces_port', 9000)
    oracle_url = f"http://{oracle_host}:{oracle_port}"

    # Log the input shape and number of classes before passing to CMdtunerSelector
    current_input_shape = (input_sequence_length, data_X.shape[2])
    current_num_classes = output_sequence_length
    logger.info(f"Passing input_shape: {current_input_shape} and num_classes: {current_num_classes} to CMdtunerSelector.")

    # Instantiate and run tuner
    dtuner_selector = CMdtunerSelector(
        backend=current_backend, # Use the dynamically determined backend
        tuner_id="tsneuropredict1",  # or None if dynamic
        oracle_url=oracle_url,  # Pass the correctly formed oracle URL
        is_chief=True, # Explicitly mark as chief
        # Any additional kwargs needed for tuning
        hypermodel_params=all_params, # Pass all_params
        dataset_params=params_dict.get("dataset_params", {}),
        base_path=params_dict.get("base_path"), # Use the correct base_path (mp_glob_base_log_path)
        model_id=params_dict.get("model_id", "tsneuromodel_1"),
        # Pass the desired model save directory to the tuner selector
        model_save_dir=Path(base_params.get('mp_glob_base_log_path')) / "tsneuromodel_1" / "saved_models",
        train_data=train_dataset, # Pass train_dataset (will be None if TF)
        val_data=val_dataset, # Pass val_dataset (will be None if TF)
        input_shape=current_input_shape, # Pass the actual input shape
        num_classes=current_num_classes # Pass the actual output sequence length
    )
    logger.info(f"✅ CMdtunerSelector initialized with backend: {dtuner_selector.backend}, tuner_id: {dtuner_selector.tuner_id}")
 

    # Start the training and tuning process
    logger.info("Starting model training and tuning...")
    try:
        dtuner_selector.run()
        logger.info("✅ Model training and tuning completed.")
    except Exception as e:
        logger.error(f"❌ Error during model training and tuning: {e}", exc_info=True)
        sys.exit(1)

    # Final Evaluation and Model Saving
    logger.info("Retrieving best model for final evaluation and saving...")
    best_model = dtuner_selector.get_best_model()

    if best_model:
        logger.info("Best model retrieved successfully. Proceeding with evaluation and saving.")

        # Prepare evaluation data (assuming data_X_test and data_y_test are already scaled and prepped)
        # If your 'evaluate_model' expects raw data, adjust this.
        # For forecasting, we often predict on a portion of the *original* data or a new unseen window.
        # Let's assume we want to predict on the entire `data_X` for visualization purposes.
        # You might want to adjust this to `data_X_test` if you only want to plot test set predictions.

        if dtuner_selector.backend == "tensorflow":
            logger.info("Running final TensorFlow model evaluation...")
            # For TensorFlow, predict directly using the best_model
            predictions = best_model.predict(data_X)
            # Flatten predictions if they are (N, 1) to (N,) for plotting
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()

            # The evaluate_model method is part of the tuner, not the selector directly.
            # Assuming the tuner has access to the evaluation data.
            # For this example, we'll call it directly on the best_model with the test data.
            # If CMdtuner has an evaluate_model method that takes X_test, y_test:
            # eval_results = dtuner_selector.tuner.evaluate_model(best_model, X_test, y_test)
            # For now, let's just calculate metrics here for simplicity:
            test_predictions_tf = best_model.predict(X_test)
            mse_tf = mean_squared_error(y_test, test_predictions_tf)
            mae_tf = mean_absolute_error(y_test, test_predictions_tf)
            r2_tf = r2_score(y_test, test_predictions_tf)
            eval_results = {'mse': mse_tf, 'mae': mae_tf, 'r2': r2_tf}
            logger.info(f"Final TensorFlow Evaluation Results: {eval_results}")

            # Save the TensorFlow model
            model_save_path = dtuner_selector.kwargs.get("model_save_dir") / "tensorflow_best_model.keras"
            logger.info(f"Saving best TensorFlow model to: {model_save_path}")
            try:
                best_model.save(model_save_path)
                logger.info(f"✅ TensorFlow model saved to {model_save_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save TensorFlow model: {e}", exc_info=True)

        elif dtuner_selector.backend == "pytorch":
            logger.info("Running final PyTorch model evaluation...")
            # For PyTorch, `evaluate_model` runs inference and computes loss.
            # We need to explicitly get predictions for plotting.
            best_model.eval() # Set to evaluation mode
            
            # Use original data_X and data_y for full forecast plot if desired
            # Or use X_test, y_test for test set only
            full_dataset = TensorDataset(torch.from_numpy(data_X).float(), torch.from_numpy(data_y).float())
            full_loader = DataLoader(full_dataset, batch_size=app_params.get('mp_app_batch_size', 32), shuffle=False)

            predictions_list = []
            with torch.no_grad():
                for inputs, _ in full_loader:
                    inputs = inputs.to(best_model.device) # Assuming model.device is set
                    outputs = best_model(inputs)
                    predictions_list.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions_list).flatten()
            
            # Assuming CMdtunerTorch has an evaluate_model method
            # For now, let's calculate metrics here for simplicity:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(best_model.device)
            y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.float32).to(best_model.device)
            
            with torch.no_grad():
                test_predictions_pt = best_model(X_test_tensor).cpu().numpy()
            
            mse_pt = mean_squared_error(y_test, test_predictions_pt)
            mae_pt = mean_absolute_error(y_test, test_predictions_pt)
            r2_pt = r2_score(y_test, test_predictions_pt)
            eval_results = {'mse': mse_pt, 'mae': mae_pt, 'r2': r2_pt}
            logger.info(f"Final PyTorch Evaluation Results: {eval_results}")

            # Save the PyTorch model
            model_save_path = dtuner_selector.kwargs.get("model_save_dir") / "pytorch_best_model.pth"
            logger.info(f"Saving best PyTorch model to: {model_save_path}")
            try:
                torch.save(best_model.state_dict(), model_save_path)
                logger.info(f"✅ PyTorch model state dict saved to {model_save_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save PyTorch model: {e}", exc_info=True)

        # --- FORECAST AND PLOT SECTION ---
        logger.info("📈 Generating forecast plot...")
        try:
            plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
            plt.figure(figsize=(15, 7))

            # --- IMPORTANT: Ensure data_y and predictions are inverse-transformed if they were scaled for training ---
            # You need access to the `scaler_y` object used during preprocessing.
            # For this example, let's assume `data_y` is your original target prices and `predictions` are the raw model outputs.
            # If `data_y` was scaled (which it is by preprocess_data), you NEED to inverse transform it.
            
            # Inverse transform data_y and predictions
            # Reshape to 2D array if they are 1D (e.g., (N,) to (N,1)) for inverse_transform
            original_actual_prices = scaler_y.inverse_transform(data_y.reshape(-1, 1)).flatten()
            predicted_prices = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # If you have original timestamps or indices, use them. Otherwise, use simple range.
            time_indices = np.arange(len(original_actual_prices))

            plt.plot(time_indices, original_actual_prices, label='Actual Price', color='blue', alpha=0.7)
            plt.plot(time_indices, predicted_prices, label='Predicted Price', color='red', linestyle='--', alpha=0.7)

            plt.title(f'Forecast Price vs. Actual Price Over Time Window ({symbol})')
            plt.xlabel('Time Step / Index')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            plot_dir = dtuner_selector.kwargs.get("model_save_dir") / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"{dtuner_selector.backend}_forecast_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            logger.info(f"✅ Forecast plot saved to {plot_path}")

            # Optionally display the plot (might block execution in some environments)
            # plt.show() # Uncomment if you want to see the plot immediately

        except Exception as e:
            logger.error(f"❌ Error generating or saving forecast plot: {e}", exc_info=True)
        # --- END FORECAST AND PLOT SECTION ---

        # ONNX conversion for TensorFlow models
        # This try-except block has to be inside the if best_model block
        if dtuner_selector.backend == "tensorflow" and app_params.get('mp_app_enable_onnx_conversion', False):
            logger.info("Attempting ONNX conversion for TensorFlow model...")
            try:
                # Assuming best_model is a tf.keras.Model
                input_signature = [tf.TensorSpec((None, input_sequence_length, data_X.shape[2]), tf.float32, name="input")]
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=13)
                
                onnx_save_path = dtuner_selector.kwargs.get("model_save_dir") / "tensorflow_best_model.onnx"
                with open(onnx_save_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"✅ ONNX model saved to {onnx_save_path}")

                # Verify ONNX model
                onnx_model_loaded = onnx.load(onnx_save_path)
                checker.check_model(onnx_model_loaded)
                logger.info("✅ ONNX model check successful.")

                # Test ONNX Runtime inference
                ort_session = ort.InferenceSession(str(onnx_save_path))
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name

                # Use a sample from test data for ONNX inference test
                test_input = X_test[0:1].astype(np.float32) # Get first sample, ensure float32
                if test_input.ndim == 2: # If it's a single sequence without batch, add batch dim
                    test_input = np.expand_dims(test_input, axis=0)

                ort_outs = ort_session.run([output_name], {input_name: test_input})
                logger.info(f"✅ ONNX Runtime inference test successful. Output shape: {ort_outs[0].shape}")

            except ImportError:
                logger.warning("tf2onnx, onnx, or onnxruntime not installed. Skipping ONNX conversion/verification.")
            except Exception as e:
                logger.error(f"❌ Failed to convert or verify ONNX model: {e}", exc_info=True)
            finally:
                pass
    else:
        logger.info("Skipping final evaluation, model saving, and plotting as no best model was found.")

    logger.info("🏁 tsNeuroPredictWinMql_chief.py finished.")


if __name__ == "__main__":
    # Ensure MetaTrader5 is initialized and finalized with authentication details
    logger.info("PARAM HEADER: MP_APP_BROKER: %s", app_params.get('mp_app_broker'))
    broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
    mqqlobj = broker_config.run_mql_login()
    if mqqlobj is True:
        logger.info("Successfully logged in to MetaTrader 5.")
    else:
        logger.info("Failed to login. Error code: %s", mqqlobj)
        sys.exit(1) # Exit if login fails

    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
    finally:
        # It's good practice to ensure MT5 is shut down properly
        try:
            mt5.shutdown()
            logger.info("MetaTrader 5 connection shut down.")
        except Exception as e:
            logger.warning(f"MetaTrader 5 shutdown failed: {e}")
        logger.info("Application process completed.")
