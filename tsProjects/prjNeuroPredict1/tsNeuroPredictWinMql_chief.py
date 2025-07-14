# filename: tsNeuroPredictWinMql_chief.py
#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                    tsNeuroPredictWinMql_chief.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
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

# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

# Import torch and related modules for PyTorch backend
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import MetaTrader5
import MetaTrader5 as mt5

# Custom modules
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr # Corrected import: Changed CEnvMgr to CMqlEnvMgr
from tsMqlConnect import CMqlBrokerConfig  # Ensure tsMqlConnect.py exists and CMqlBrokerConfig is in it
from tsMqlPlotService import PlottingService # UNCOMMENTED THIS LINE

# Import the CMdtunerSelector
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlLogService import CMLogServiceSetup

# --- Global Configuration Loading (for module-level access if needed) ---
# These are loaded here for module-level constants and default values.
# Actual runtime parameters will be passed to the main task function.
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
    role_hint='chief_module_init', # A distinct role hint for the module's own logger
    loglevel=app_params_global.get('LOGLEVEL', 'INFO').upper()
)
logger = logging.getLogger(__name__)
logger.info(f"Chief module-level logging initialized. Log level: {app_params_global.get('LOGLEVEL', 'INFO').upper()}")


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


def run_chief_process_task(tuner_id: str, oracle_url: str, is_chief: bool,
                           app_params: dict, tune_params: dict, base_params: dict):
    """
    Main task function for the chief process.
    This function will be called by multiprocessing.Process.
    """
    # Re-initialize logging specifically for this process's execution context
    # This ensures logs from this process go to the correct, unique file.
    log_level = app_params.get('LOGLEVEL', 'INFO').upper()
    CMLogServiceSetup.initialize_logging(
        app_params=app_params,
        tune_params=tune_params,
        base_params=base_params,
        role_hint=tuner_id, # Use tuner_id (e.g., 'chief') as role hint for unique log file
        loglevel=log_level,
        logfile=f"{tuner_id}_{app_params.get('xerces_logfile', 'tsneuropredict_app.log')}"
    )
    process_logger = logging.getLogger(__name__) # Get logger for this specific process
    process_logger.info(f"Chief process task started. Tuner ID: {tuner_id}, Is Chief: {is_chief}, Oracle URL: {oracle_url}")
    process_logger.info(f"Chief process LOGDIR: {base_params.get('mp_glob_base_log_path')}")


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
            is_chief=is_chief,
            train_data=(train_data_x, train_data_y),
            val_data=(val_data_x, val_data_y),
            input_shape=input_shape,
            oracle_directory=str(Path(base_params.get('mp_glob_base_log_path')) / "keras_tuner_chief_data"),
            model_save_dir=Path(base_params.get('mp_glob_base_log_path')) / "tsneuromodel_1" / "saved_models",
            app_params=app_params, # Pass app_params for ModelCheckpoint in TunerMod
            tune_params=tune_params # Pass tune_params for tuner configuration
        )

        process_logger.info(f"Chief {tuner_id} starting its trial execution loop...")
        tuner_selector.run()
        process_logger.info(f"Chief {tuner_id} trial execution completed.")

        # 4. Get Best Model from Oracle (after all trials are expected to be completed)
        process_logger.info("Attempting to retrieve the best model from the Oracle Server...")
        best_model = tuner_selector.get_best_model()

        if best_model:
            process_logger.info("✅ Best model retrieved successfully.")

            # 5. Evaluate the Best Model
            process_logger.info("Evaluating the best model on validation data...")
            if backend_for_log == 'tensorflow' or backend_for_log == 'keras':
                y_pred_val = best_model.predict(val_data_x)
            elif backend_for_log == 'pytorch':
                best_model.eval()
                with torch.no_grad():
                    val_data_x_tensor = torch.tensor(val_data_x, dtype=torch.float32).to(tuner_selector.tuner.device)
                    y_pred_val = best_model(val_data_x_tensor).cpu().numpy()
            else:
                raise ValueError("Unsupported backend for evaluation.")

            mse = mean_squared_error(val_data_y, y_pred_val)
            mae = mean_absolute_error(val_data_y, y_pred_val)
            r2 = r2_score(val_data_y, y_pred_val)

            process_logger.info(f"Best Model Evaluation on Validation Data:")
            process_logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
            process_logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
            process_logger.info(f"  R-squared (R2): {r2:.4f}")

            # 6. Save Final Best Model (Chief's responsibility)
            model_save_path = tuner_selector.get_model_dir()
            model_save_path.mkdir(parents=True, exist_ok=True)
            final_model_path = model_save_path / f"final_best_model_{backend_for_log}"

            try:
                if backend_for_log == 'tensorflow' or backend_for_log == 'keras':
                    best_model.save(str(final_model_path))
                    process_logger.info(f"✅ Final best TensorFlow/Keras model saved to {final_model_path}")
                elif backend_for_log == 'pytorch':
                    torch.save(best_model.state_dict(), str(final_model_path.with_suffix('.pth')))
                    process_logger.info(f"✅ Final best PyTorch model state dict saved to {final_model_path.with_suffix('.pth')}")
                
                import joblib
                scaler_path = model_save_path / "scaler.joblib"
                joblib.dump(StandardScaler(), scaler_path)
                process_logger.info(f"Scaler saved to {scaler_path}")

            except Exception as e:
                process_logger.error(f"❌ Failed to save the final best model: {e}", exc_info=True)

            # 7. Plotting (Chief's responsibility)
            try:
                plotting_service = PlottingService(log_dir=Path(base_params.get('mp_glob_base_log_path')))
                
                y_pred_val_flat = y_pred_val.flatten()
                val_data_y_flat = val_data_y.flatten()

                plotting_service.plot_predictions(val_data_y_flat, y_pred_val_flat, "Chief - Actual vs. Predicted (Validation Set)", f"predictions_chief_{backend_for_log}.png")
                residuals = val_data_y_flat - y_pred_val_flat
                plotting_service.plot_residuals(residuals, "Chief - Residuals Plot (Validation Set)", f"residuals_chief_{backend_for_log}.png")
                plotting_service.plot_scatter(val_data_y_flat, y_pred_val_flat, "Chief - Actual vs. Predicted Scatter", f"scatter_chief_{backend_for_log}.png")

                process_logger.info("✅ Plots generated successfully.")
            except Exception as e:
                process_logger.error(f"❌ Error during plotting: {e}", exc_info=True)

            # 8. ONNX Conversion (Chief's responsibility, for TensorFlow/Keras only)
            if backend_for_log == 'tensorflow' or backend_for_log == 'keras':
                process_logger.info("Attempting ONNX conversion and verification for TensorFlow/Keras model...")
                try:
                    import tf2onnx
                    import onnx
                    from onnx import checker
                    import onnxruntime as ort

                    onnx_model_path = model_save_path / f"final_best_model_{backend_for_log}.onnx"
                    input_signature = [tf.TensorSpec(best_model.inputs[0].shape, best_model.inputs[0].dtype, name="input_1")]
                    onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=13)
                    with open(onnx_model_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                    process_logger.info(f"✅ ONNX model saved to {onnx_model_path}")

                    onnx_model = onnx.load(onnx_model_path)
                    checker.check_model(onnx_model)
                    process_logger.info("✅ ONNX model verified successfully.")

                    ort_session = ort.InferenceSession(str(onnx_model_path))
                    ort_inputs = {ort_session.get_inputs()[0].name: val_data_x.astype(np.float32)}
                    ort_outs = ort_session.run(None, ort_inputs)
                    np.testing.assert_allclose(y_pred_val, ort_outs[0], rtol=1e-3, atol=1e-3)
                    process_logger.info("✅ ONNX Runtime output matches TensorFlow/Keras output.")
                    process_logger.info(f"ONNX output shape: {ort_outs[0].shape}")

                except ImportError:
                    process_logger.warning("tf2onnx, onnx, or onnxruntime not installed. Skipping ONNX conversion/verification.")
                except Exception as e:
                    process_logger.error(f"❌ Failed to convert or verify ONNX model: {e}", exc_info=True)
                finally:
                    pass
        else:
            process_logger.info("Skipping final evaluation, model saving, and plotting as no best model was found.")

    except Exception as e:
        process_logger.critical(f"Unhandled exception in chief process task: {e}", exc_info=True)
        sys.exit(1) # Exit with error code

    finally:
        # Ensure MT5 is shut down properly for this process
        try:
            mt5.shutdown()
            process_logger.info("MetaTrader 5 connection shut down.")
        except Exception as e:
            process_logger.warning(f"Error during MT5 shutdown in chief process: {e}")

    process_logger.info("🏁 Chief process task finished.")


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
    tuner_id = os.environ.get("TUNER_ID", "chief_standalone")
    oracle_url = os.environ.get("ORACLE_URL")
    is_chief = os.environ.get("IS_CHIEF", "true").lower() == "true"

    # Re-load parameters from environment for this specific process
    # This is important because the child process has its own environment.
    mql_overrides_child = CMqlOverrides()
    all_params_child = mql_overrides_child.env.all_params()
    app_params_child = all_params_child.get("app", {})
    tune_params_child = all_params_child.get('mltune', {})
    base_params_child = all_params_child.get("base", {})

    if not oracle_url:
        logging.getLogger(__name__).critical("ORACLE_URL environment variable not set in child process. Exiting.")
        sys.exit(1)

    run_chief_process_task(tuner_id, oracle_url, is_chief,
                           app_params_child, tune_params_child, base_params_child)
