# filename: tsNeuroPredictWinMql_chief.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsNeuroPredictWinMql_chief.py
Description: The chief process for distributed machine learning tuning.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.4.0
License: MIT License
"""
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

# Add this import for type hints
from typing import Dict, Any
# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

# Import torch and related modules for PyTorch backend
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx # Added for PyTorch ONNX export

# Import MetaTrader5
import MetaTrader5 as mt5

# Custom modules
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr # Corrected import: Changed CEnvMgr to CMqlEnvMgr
from tsMqlConnect import CMqlBrokerConfig  # Ensure tsMqlConnect is importable
from tsMqlLogService import CMLogServiceSetup # Import the centralized logging setup
from tsMqlPlotService import PlottingService # Import PlottingService
from tsMqlMLProcess import CDMLProcess # Import CDMLProcess

# Import the distributed tuner selector
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient

# --- START OF NUMPY 2.0 COMPATIBILITY PATCH FOR TF2ONNX ---
# This section attempts to patch numpy.cast for tf2onnx compatibility with NumPy 2.0+.
# This is a temporary workaround for an issue in older tf2onnx versions that
# try to use `np.cast`, which was removed in NumPy 2.0.
# The proper solution is to update tf2onnx to a version compatible with NumPy 2.0.
TF2ONNX_NUMPY_PATCHED = False
try:
    if tuple(map(int, np.__version__.split('.'))) >= (2, 0, 0):
        if not hasattr(np, 'cast'):
            # Define a simple np.cast that mimics the old behavior using np.asarray
            # This is a minimal patch to allow tf2onnx to import without error.
            def _np_cast_patch(arr, dtype):
                return np.asarray(arr, dtype=dtype)
            np.cast = _np_cast_patch
            TF2ONNX_NUMPY_PATCHED = True
            logging.getLogger(__name__).warning(
                "NumPy 2.0+ detected. Temporarily patched `np.cast` for tf2onnx compatibility. "
                "Consider upgrading tf2onnx for a permanent fix."
            )
except Exception as e:
    logging.getLogger(__name__).error(f"Error applying NumPy 2.0 compatibility patch for tf2onnx: {e}")
# --- END OF NUMPY 2.0 COMPATIBILITY PATCH ---

# Onnx and tf2onnx imports (kept for imports, but conversion logic removed from worker main)
try:
    import tf2onnx
    import onnx
    from onnx import checker
    import onnxruntime as ort
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    pass


# Setup logging for the chief process
mql_overrides_init = CMqlOverrides()
all_params_init = mql_overrides_init.env.all_params()
app_params_init = all_params_init.get("app", {})
tune_params_init = all_params_init.get('mltune', {})
base_params_init = all_params_init.get("base", {})

backend_for_log_init = os.environ.get('BACKEND', tune_params_init.get('backend', 'pytorch'))

CMLogServiceSetup.initialize_logging(
    app_params=app_params_init,
    tune_params=tune_params_init,
    base_params=base_params_init,
    role_hint='chief',
    loglevel='INFO',
    logfile='tsneuropredict_app.log',
    backend=backend_for_log_init
)
logger = logging.getLogger(__name__)


def run_chief_process_task(tuner_id: str, oracle_url: str, is_chief: bool,
                           app_params: Dict[str, Any], tune_params: Dict[str, Any], base_params: Dict[str, Any]):
    """
    The main task for the chief process.
    It initializes the OracleClient, runs the tuner, retrieves the best model,
    evaluates it, and performs ONNX conversion if enabled.
    """
    logger.info(f"Chief {tuner_id} process task started. Is Chief: {is_chief}")

    # Initialize OracleClient
    oracle_client = OracleClient(oracle_url=oracle_url)
    logger.info(f"OracleClient initialized for chief, connecting to {oracle_url}")

    # --- Start: Load and Process real data using CDMLProcess ---
    logger.info("Attempting to load and process real data using CDMLProcess...")
    x_data, y_data = np.array([]), np.array([]) # Initialize as empty arrays

    try:
        # Initialize CDMLProcess with all_params
        ml_processor = CDMLProcess(all_params=all_params_init)
        
        # Load and prepare data using the new method
        x_data, y_data = ml_processor.load_and_prepare_data(app_params_init, tune_params_init, base_params_init)

        # Check if data loading was successful and data is not empty
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Loaded real data (x_data or y_data) is empty after processing.")

        # Dynamically determine input_shape from the loaded data
        if x_data.ndim == 3:
            input_shape = x_data.shape[1:] # For sequence data (samples, timesteps, features)
        elif x_data.ndim == 2:
            input_shape = (x_data.shape[1],) # For flat features (samples, features)
        else:
            raise ValueError(f"Unsupported x_data dimensions: {x_data.ndim}. Expected 2D or 3D array.")

        logger.info(f"✅ Real data loaded and processed successfully. x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")
        logger.info(f"Dynamically determined input_shape for model: {input_shape}")

    except Exception as e:
        logger.critical(f"❌ Failed to load and process real data using CDMLProcess: {e}", exc_info=True)
        logger.critical("Using dummy data as fallback. Model performance will be meaningless.")
        # Fallback to dummy data if real data loading fails
        input_shape = (10,) # Default dummy input shape
        num_samples = 1000
        x_data = np.random.rand(num_samples, input_shape[0]).astype(np.float32)
        y_data = np.random.rand(num_samples, 1).astype(np.float32) * 100 # Dummy labels
    # --- End: Load and Process real data using CDMLProcess ---

    # Split data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Determine model save directory
    model_save_dir = Path(base_params.get('mp_glob_base_log_path')) / app_params.get('mp_app_model_id', 'default_model') / "saved_models"
    model_save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Initialize the Tuner Selector based on the backend
    backend = tune_params.get('backend', 'tensorflow')
    logger.info(f"Initializing CMdtunerSelector with backend: {backend}")

    # Pass the actual data and input_shape to the tuner
    tuner_selector = CMdtunerSelector(
        backend=backend,
        tuner_id=tuner_id,
        oracle_client=oracle_client,
        is_chief=is_chief,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        input_shape=input_shape, # Pass the dynamically determined input_shape
        model_save_dir=model_save_dir,
        oracle_directory=Path(tune_params.get('tuner_dir', base_params.get('mp_glob_base_log_path') / "keras_tuner_data")),
        app_params=app_params,
        tune_params=tune_params
    )

    # Run the tuner (this will block until all trials are processed or max_trials reached)
    logger.info("Chief chief trial execution starting.")
    tuner_selector.run()
    logger.info("Chief chief trial execution completed.")

    # Retrieve the best model
    logger.info("Attempting to retrieve the best model from the Oracle Server...")
    best_model = tuner_selector.get_best_model()

    if best_model:
        logger.info("✅ Best model retrieved successfully.")

        # Save the final best model conditionally based on backend
        final_model_name = f"final_best_model_{backend}"
        try:
            if backend == 'tensorflow':
                final_model_path = model_save_dir / f"{final_model_name}.keras"
                best_model.save(str(final_model_path))
                logger.info(f"✅ Final best TensorFlow model saved to: {final_model_path}")
            elif backend == 'pytorch':
                final_model_path = model_save_dir / f"{final_model_name}.pth"
                torch.save(best_model.state_dict(), final_model_path)
                logger.info(f"✅ Final best PyTorch model state_dict saved to: {final_model_path}")
            else:
                logger.warning(f"Model saving not implemented for backend: {backend}")
        except Exception as e:
            logger.error(f"❌ Failed to save the final best model: {e}", exc_info=True)

        # Evaluate the best model on validation data
        logger.info("Evaluating the best model on validation data...")
        loss, mae, y_pred, y_true = None, None, None, None # Initialize variables

        if backend == 'tensorflow':
            # Evaluate directly using Keras evaluate method
            loss, _ = best_model.evaluate(x_val, y_val, verbose=0) # _ to discard other metrics if any
            y_pred = best_model.predict(x_val).flatten()
            y_true = y_val.flatten() # Ensure y_true is also flattened for consistent comparison
            mae = mean_absolute_error(y_true, y_pred) # Calculate MAE explicitly
        elif backend == 'pytorch':
            # Convert validation data to tensors and move to device
            val_tensor_x = torch.tensor(x_val, dtype=torch.float32).to(tuner_selector.tuner.device)
            val_tensor_y = torch.tensor(y_val, dtype=torch.float32).to(tuner_selector.tuner.device)
            val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
            val_loader = DataLoader(val_dataset, batch_size=tuner_selector.tuner.batch_size, shuffle=False)

            best_model.eval()
            all_preds = []
            all_targets = []
            total_loss = 0
            criterion = nn.MSELoss()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = best_model(X_batch)
                    loss_batch = criterion(outputs, y_batch)
                    total_loss += loss_batch.item()
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())
            loss = total_loss / len(val_loader)
            y_pred = np.array(all_preds)
            y_true = np.array(all_targets)
            mae = mean_absolute_error(y_true, y_pred)
        else:
            logger.warning(f"Evaluation not implemented for backend: {backend}")

        if loss is not None and mae is not None and y_true is not None and y_pred is not None:
            # Ensure y_true and y_pred have consistent shapes before calculating R2 score
            # Reshape to 1D arrays if they are not already, for consistent behavior with sklearn metrics
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            r2 = r2_score(y_true, y_pred)
            logger.info("Best Model Evaluation on Validation Data:")
            logger.info(f"  Mean Squared Error (MSE): {loss:.4f}")
            logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
            logger.info(f"  R-squared (R2): {r2:.4f}")

            # Initialize PlottingService
            plotting_service = PlottingService(Path(base_params.get('mp_glob_base_log_path')))
            
            # Log model performance analysis using the PlottingService
            plotting_service.log_model_performance(loss, mae, r2)

            # Calculate residuals
            residuals = y_true - y_pred # Use y_true for residuals

            # Generate plots
            plotting_service.plot_predictions(y_true, y_pred, f'({backend} Chief)', f'chief_{backend}')
            plotting_service.plot_residuals(residuals, f'Residuals ({backend} Chief)', f'chief_{backend}_residuals.png')
            plotting_service.plot_scatter(y_true, y_pred, f'({backend} Chief)', f'chief_{backend}')
            logger.info("✅ Plots generated successfully.")
        else:
            logger.warning("Skipping plot generation due to missing evaluation metrics.")

        # ONNX conversion and verification
        if TF2ONNX_AVAILABLE:
            if backend == 'tensorflow':
                logger.info("Attempting ONNX conversion and verification for TensorFlow/Keras model...")
                try:
                    if isinstance(best_model, tf.keras.Sequential) and not hasattr(best_model, 'output_names'):
                        best_model.output_names = ["output_1"]
                        logger.warning("Temporarily added 'output_names' attribute to Sequential model for tf2onnx compatibility.")

                    if hasattr(best_model, 'input_shape') and best_model.input_shape is not None:
                        concrete_input_shape = (1,) + best_model.input_shape[1:]
                        input_signature = [tf.TensorSpec(concrete_input_shape, tf.float32, name="input_1")]
                    else:
                        logger.warning("Model input_shape not directly available, using default (None, 10) for ONNX conversion.")
                        input_signature = [tf.TensorSpec((None, 10), tf.float32, name="input_1")]

                    onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=13)
                    
                    onnx_model_path = model_save_dir / f"best_model_{backend}.onnx"
                    with open(onnx_model_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                    logger.info(f"✅ ONNX model saved to: {onnx_model_path}")

                    onnx.checker.check_model(onnx_model)
                    logger.info("✅ ONNX model verification successful.")

                    ort_session = ort.InferenceSession(str(onnx_model_path))
                    onnx_input_name = ort_session.get_inputs()[0].name
                    onnx_output_name = ort_session.get_outputs()[0].name

                    sample_input = x_val[:1].astype(np.float32)
                    onnx_preds = ort_session.run([onnx_output_name], {onnx_input_name: sample_input})[0]
                    logger.info(f"✅ ONNX Runtime inference successful for a sample. Predictions: {onnx_preds.flatten()}")

                except Exception as e:
                    logger.error(f"❌ Failed to convert or verify ONNX model for TensorFlow/Keras: {e}", exc_info=True)
            elif backend == 'pytorch':
                logger.info("Attempting ONNX conversion and verification for PyTorch model...")
                try:
                    best_model.eval()

                    # Use tuner_selector.tuner.device to get the correct device
                    dummy_input = torch.randn(1, *input_shape, device=tuner_selector.tuner.device, dtype=torch.float32)
                    
                    onnx_model_path = model_save_dir / f"best_model_{backend}.onnx"

                    torch.onnx.export(
                        best_model,
                        dummy_input,
                        onnx_model_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                      'output': {0: 'batch_size'}}
                    )
                    logger.info(f"✅ ONNX model saved to: {onnx_model_path}")

                    onnx_model = onnx.load(onnx_model_path)
                    onnx.checker.check_model(onnx_model)
                    logger.info("✅ ONNX model verification successful.")

                    ort_session = ort.InferenceSession(str(onnx_model_path))
                    onnx_input_name = ort_session.get_inputs()[0].name
                    onnx_output_name = ort_session.get_outputs()[0].name

                    sample_input = x_val[:1].astype(np.float32)
                    onnx_preds = ort_session.run([onnx_output_name], {onnx_input_name: sample_input})[0]
                    logger.info(f"✅ ONNX Runtime inference successful for a sample. Predictions: {onnx_preds.flatten()}")

                except Exception as e:
                    logger.error(f"❌ Failed to convert or verify ONNX model for PyTorch: {e}", exc_info=True)
        else:
            logger.info("ONNX conversion skipped for non-TensorFlow backend or if tf2onnx is not available.")
    else:
        logger.error("❌ No best model found or retrieved. Skipping evaluation and ONNX conversion.")

    # Disconnect MetaTrader 5 if connected
    if mt5.initialize():
        mt5.shutdown()
        logger.info("MetaTrader 5 connection shut down.")
    else:
        logger.warning("MetaTrader 5 connection was not initialized, skipping shutdown.")


    logger.info("🏁 Chief process task finished.")