# filename: tsNeuroPredictWinMql_chief.py
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
import numpy as np # Import numpy here, before tf2onnx is potentially imported
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

# Import MetaTrader5
import MetaTrader5 as mt5

# Custom modules
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr # Corrected import: Changed CEnvMgr to CMqlEnvMgr
from tsMqlConnect import CMqlBrokerConfig  # Ensure tsMqlConnect is importable
from tsMqlLogService import CMLogServiceSetup # Import the centralized logging setup

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
    # print("tf2onnx, onnx, or onnxruntime not installed. ONNX conversion/verification will be skipped.") # Use logger instead
    pass # Let the logger handle this in the main function


# Setup logging for the chief process
# Load configuration (similar to other main scripts)
mql_overrides_init = CMqlOverrides()
all_params_init = mql_overrides_init.env.all_params()
app_params_init = all_params_init.get("app", {})
tune_params_init = all_params_init.get('mltune', {})
base_params_init = all_params_init.get("base", {})

# Determine the backend from environment, default to 'pytorch'
backend_for_log_init = os.environ.get('BACKEND', tune_params_init.get('backend', 'pytorch'))

CMLogServiceSetup.initialize_logging(
    app_params=app_params_init,
    tune_params=tune_params_init,
    base_params=base_params_init,
    role_hint='chief',
    loglevel='INFO',
    logfile='tsneuropredict_app.log', # Main log file for the application
    backend=backend_for_log_init # Re-added backend parameter as it is now in initialize_logging signature
)
logger = logging.getLogger(__name__)

# Define a plotting service class (can be moved to a separate module if it grows)
class PlottingService:
    def __init__(self, log_dir: Path):
        self.plot_dir = log_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PlottingService initialized. Plots will be saved to: {self.plot_dir}")

    def plot_predictions(self, y_true, y_pred, title_suffix, filename_suffix):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Actual vs Predicted {title_suffix}')
        plt.xlabel('Time/Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"predictions_{filename_suffix}.png")
        plt.close()
        logger.info(f"Plot saved successfully: {self.plot_dir / f'predictions_{filename_suffix}.png'}")

    def plot_residuals(self, y_true, y_pred, title_suffix, filename_suffix):
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 6))
        plt.hist(residuals, bins=50)
        plt.title(f'Residuals Distribution {title_suffix}')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"residuals_{filename_suffix}.png")
        plt.close()
        logger.info(f"Plot saved successfully: {self.plot_dir / f'residuals_{filename_suffix}.png'}")

    def plot_scatter(self, y_true, y_pred, title_suffix, filename_suffix):
        plt.figure(figsize=(8, 8))
        sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Scatter Plot {title_suffix}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"scatter_{filename_suffix}.png")
        plt.close()
        logger.info(f"Plot saved successfully: {self.plot_dir / f'scatter_{filename_suffix}.png'}")


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

    # Dummy data for demonstration (replace with actual data loading)
    # In a real scenario, you would load your preprocessed data here.
    # For now, let's create some random data that matches the expected input_shape.
    input_shape = (10,) # Example: 10 features per time step
    num_samples = 1000
    x_data = np.random.rand(num_samples, input_shape[0]).astype(np.float32)
    y_data = np.random.rand(num_samples, 1).astype(np.float32) * 100 # Dummy labels

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
        input_shape=input_shape,
        model_save_dir=model_save_dir,
        oracle_directory=Path(tune_params.get('tuner_dir', base_params.get('mp_glob_base_log_path') / "keras_tuner_data")),
        app_params=app_params, # Pass app_params
        tune_params=tune_params # Pass tune_params
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

        # Save the final best model
        final_model_name = f"final_best_model_{backend}"
        # Ensure a proper file extension is added
        final_model_path = model_save_dir / f"{final_model_name}.keras" # Changed to .keras extension
        try:
            best_model.save(str(final_model_path))
            logger.info(f"✅ Final best model saved to: {final_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save the final best model: {e}", exc_info=True)

        # Evaluate the best model on validation data
        logger.info("Evaluating the best model on validation data...")
        if backend == 'tensorflow':
            loss, mae = best_model.evaluate(x_val, y_val, verbose=0)
            y_pred = best_model.predict(x_val).flatten()
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
            loss, mae, y_pred = None, None, None

        if loss is not None and mae is not None:
            r2 = r2_score(y_val, y_pred)
            logger.info("Best Model Evaluation on Validation Data:")
            logger.info(f"  Mean Squared Error (MSE): {loss:.4f}")
            logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
            logger.info(f"  R-squared (R2): {r2:.4f}")

            # Generate plots
            plotting_service = PlottingService(Path(base_params.get('mp_glob_base_log_path')))
            plotting_service.plot_predictions(y_val.flatten(), y_pred, f'({backend} Chief)', f'chief_{backend}')
            plotting_service.plot_residuals(y_val.flatten(), y_pred, f'({backend} Chief)', f'chief_{backend}')
            plotting_service.plot_scatter(y_val.flatten(), y_pred, f'({backend} Chief)', f'chief_{backend}')
            logger.info("✅ Plots generated successfully.")
        else:
            logger.warning("Skipping plot generation due to missing evaluation metrics.")

        # ONNX conversion and verification
        if TF2ONNX_AVAILABLE and backend == 'tensorflow':
            logger.info("Attempting ONNX conversion and verification for TensorFlow/Keras model...")
            try:
                # Workaround for tf2onnx expecting 'output_names' on Sequential models
                # Check if it's a Sequential model and if it lacks the 'output_names' attribute
                if isinstance(best_model, tf.keras.Sequential) and not hasattr(best_model, 'output_names'):
                    # Assign a default output name to satisfy tf2onnx's internal check
                    # Assuming a single output for typical Sequential models
                    best_model.output_names = ["output_1"]
                    logger.warning("Temporarily added 'output_names' attribute to Sequential model for tf2onnx compatibility.")

                # Infer input shape from the model directly
                if hasattr(best_model, 'input_shape') and best_model.input_shape is not None:
                    concrete_input_shape = (1,) + best_model.input_shape[1:]
                    input_signature = [tf.TensorSpec(concrete_input_shape, tf.float32, name="input_1")]
                else:
                    logger.warning("Model input_shape not directly available, using default (None, 10) for ONNX conversion.")
                    input_signature = [tf.TensorSpec((None, 10), tf.float32, name="input_1")]

                # Now call from_keras. The previous workaround should prevent the AttributeError.
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=13)
                
                onnx_model_path = model_save_dir / f"best_model_{backend}.onnx"
                with open(onnx_model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"✅ ONNX model saved to: {onnx_model_path}")

                # Verify ONNX model
                onnx.checker.check_model(onnx_model)
                logger.info("✅ ONNX model verification successful.")

                # Test ONNX model with ONNX Runtime
                ort_session = ort.InferenceSession(str(onnx_model_path))
                onnx_input_name = ort_session.get_inputs()[0].name
                onnx_output_name = ort_session.get_outputs()[0].name

                # Use a subset of validation data for ONNX inference
                # Ensure sample_input matches the concrete_input_shape used for export
                sample_input = x_val[:1].astype(np.float32) # Use batch size 1 for testing
                onnx_preds = ort_session.run([onnx_output_name], {onnx_input_name: sample_input})[0]
                logger.info(f"✅ ONNX Runtime inference successful for a sample. Predictions: {onnx_preds.flatten()}")

            except Exception as e:
                logger.error(f"❌ Failed to convert or verify ONNX model: {e}", exc_info=True)
        elif backend == 'pytorch':
            logger.info("ONNX conversion for PyTorch models is not yet implemented in this chief script.")
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
