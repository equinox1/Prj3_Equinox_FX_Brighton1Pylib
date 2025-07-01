#!/usr/bin/env python3
# +------------------------------------------------------------------+\
# |                                    tsNeuroPredictWinMql_chief.py |\
# |                                                    Tony Shepherd |\
# |                                    https://www.xercescloud.co.uk |\
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
import MetaTrader5 as mt5

# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

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
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer # For running server locally if needed
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector


# Keras Tuner components for manual trial management
from keras_tuner.engine.trial import TrialStatus

# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" # Suppress TensorFlow warnings, only show errors

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))



# Initialize CMqlSetup to get configuration, including precision
# Dynamically determine num_cores and num_threads for optimal performance.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel=app_params.get('LOGLEVEL', 'INFO'), # Get loglevel from app_params
    warn='ignore', # Default or get from params if available
    precision=app_params.get('TF_PRECISION', 'mixed_float16'), # Default or get from params
    tfdebug=app_params.get('TFDEBUG', False),
    num_cores=_estimated_physical_cores,
    num_threads=_logical_cores # Can be set higher if testing proves beneficial
)


# Set mixed precision policy if not already set globally by CMqlSetup
policy = mixed_precision.Policy(setup_config.precision)
mixed_precision.set_global_policy(policy)
logger.info(f"✨ Global mixed precision policy set to: {mixed_precision.global_policy().compute_dtype}")

# Model and Tuner Configuration
MODEL_NAME = tune_params.get('ml_model_name', 'tsneuromodel')
MODEL_DIR = Path(base_params.get('mp_glob_sub_ml_src_modeldata', 'tsModelData')) # Use base_params for model data path
PROJECT_PATH = MODEL_DIR / MODEL_NAME
PROJECT_PATH.mkdir(parents=True, exist_ok=True) # Ensure project directory exists

TUNER_ID_CHIEF = app_params.get('tuner_id_chief', 'chief')
NUM_TRIALS_CHIEF = tune_params.get('num_trials', 64)
# Chief determines overwrite behavior for the *entire* tuning process
OVERWRITE = tune_params.get('overwrite', True)

# Oracle Server configuration
oracle_server_enabled = tune_params.get('tunertype', 'local') == 'remote'
xerces_server = app_params.get('xerces_server', '127.0.0.1')
xerces_port = app_params.get('xerces_port', 9000)
oracle_url = f"http://{xerces_server}:{xerces_port}"
LOGDIR = app_params.get('LOGDIR', 'Logdir')
LOGDIR = Path(LOGDIR)  # Convert LOGDIR to a Path object
oracle_full_path = LOGDIR / "oracle_server"

# ----------------------------
# Main Logic
# ----------------------------
def main():
    logger.info(f"Chief {TUNER_ID_CHIEF} started. Kicking off data loading and processing.")

    # The 'all_params' dictionary is already loaded at the global scope.
    # We can directly use 'all_params' instead of re-initializing CMqlEnvMgr.
    
    # 1. Data Loading
    # Extract parameters for CDataLoader from all_params
    primary_symbol = app_params.get('mp_app_primary_symbol', 'EURUSD')
    timeframe_str = app_params.get('mp_app_timeframe', 'mt5.TIMEFRAME_H4')
    
    # Dynamically resolve timeframe string to mt5 constant
    # This ensures that the correct mt5.TIMEFRAME_H4 (or other) is passed.
    try:
        timeframe = getattr(mt5, timeframe_str.split('.')[-1])
        logger.info(f"Resolved timeframe: {timeframe_str} to MT5 constant {timeframe}")
    except AttributeError:
        logger.error(f"Invalid timeframe string: {timeframe_str}. Falling back to mt5.TIMEFRAME_H4.")
        timeframe = mt5.TIMEFRAME_H4 # Fallback
    
    start_date = app_params.get('mp_app_start_date', '2023-01-01')
    end_date = app_params.get('mp_app_end_date', datetime.now().strftime('%Y-%m-%d'))
    data_path = base_params.get('mp_glob_base_data_path', 'Mql5Data')

    # Collect all kwargs for CDataLoader
    data_loader_kwargs = {
        'mp_data_rows': all_params.get('data', {}).get('mp_data_rows', 1000),
        'mp_data_rowcount': all_params.get('data', {}).get('mp_data_rowcount', 10000),
        'mp_data_loadapiticks': all_params.get('data', {}).get('mp_data_loadapiticks', True),
        'mp_data_loadapirates': all_params.get('data', {}).get('mp_data_loadapirates', True),
        'mp_data_loadfileticks': all_params.get('data', {}).get('mp_data_loadfileticks', True),
        'mp_data_loadfilerates': all_params.get('data', {}).get('mp_data_loadfilerates', True)
    }

    # FIX: Pass positional arguments explicitly
    data_loader = CDataLoader(
        symbol=primary_symbol,
        timeframe=timeframe,
        start_date_str=start_date,
        end_date_str=end_date,
        data_path=data_path,
        **data_loader_kwargs # Unpack the dictionary into keyword arguments for the rest
    )

    # CDataLoader.run_dataloader_services now returns a dictionary
    all_dfs = data_loader.run_dataloader_services() 
    logger.info(f"Data Loader services finished. Loaded DataFrames: {all_dfs.keys()}")

    # Determine which DataFrame to use based on configuration
    used_data_key = app_params.get('mp_app_cfg_usedata', 'df_file_rates') # Default to file_rates
    logger.info(f"DEBUG: In tsNeuroPredictWinMql_chief.py - used_data_key for main_df: '{used_data_key}'")
    logger.info(f"DEBUG: In tsNeuroPredictWinMql_chief.py - Keys in all_dfs received from CDataLoader: {list(all_dfs.keys())}")

    main_df = all_dfs.get(used_data_key)

    if main_df is None or main_df.empty:
        logger.error(f"❌ Main DataFrame '{used_data_key}' is not available or is empty after data loading. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded DataFrame '{used_data_key}' with shape: {main_df.shape}")
    # logger.debug(f"DataFrame head:\n{tabulate(main_df.head(), headers='keys', tablefmt='psql')}")

    # 2. Data Processing (using CDataProcess)
    # The CDataProcess class definition was found in tsMqlDataProcess.py.
    # Instantiate and use it.
    # 2. Data Processing (using CDataProcess)
    # The CDataProcess class definition was found in tsMqlDataProcess.py.
    # Instantiate and use it.
    data_processor = CDataProcess(
        df=main_df,
        all_params=all_params,  # Pass all_params as a keyword argument
        project_dir=PROJECT_PATH
    )
    processed_df = data_processor.process_data()

    if processed_df is None or processed_df.empty:
        logger.error("❌ Processed DataFrame is empty after CDataProcess. Exiting.")
        sys.exit(1)

    logger.info(f"Processed DataFrame shape after CDataProcess: {processed_df.shape}")


    # 3. Machine Learning Data Preparation (using CDMLProcess)
    # CDMLProcess will now handle feature engineering and return X, y as numpy arrays
    ml_processor = CDMLProcess(df=processed_df, all_params=all_params, project_dir=PROJECT_PATH)
    
    # Corrected: CDMLProcess.process_ml_data returns X (numpy array), input_shape, num_classes
    X_final, y_final, input_shape, num_classes = ml_processor.process_ml_data()

    # Check if the returned numpy arrays are empty
    if X_final is None or X_final.size == 0 or y_final is None or y_final.size == 0:
        logger.error("❌ Processed X or y are empty after ML processing. Exiting.")
        sys.exit(1)
    
    logger.info(f"ML Processed X shape: {X_final.shape}, Y shape: {y_final.shape}")

    # Split data into training, validation, and test sets
    random_state = tune_params.get('seed', 42)
    X_train, X_val, X_test, y_train, y_val, y_test = ml_processor.split_dataset(
        X_final, y_final, # Pass the numpy arrays
        train_size=tune_params.get('train_split', 0.7),
        val_size=tune_params.get('val_split', 0.15),
        test_size=tune_params.get('test_split', 0.15),
        random_state=random_state
    )
    logger.info(f"Data split into: Train X: {X_train.shape}, Y: {y_train.shape} | Val X: {X_val.shape}, Y: {y_val.shape} | Test X: {X_test.shape}, Y: {y_test.shape}")


    # Convert to TensorFlow Datasets using the appropriate method from CDMLProcess
    train_dataset, val_dataset, test_dataset = ml_processor.create_tf_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=tune_params.get('batch_size', 32),
        shuffle_buffer=tune_params.get('buffer_size', 1000) # Changed from buffer_size to shuffle_buffer
    )

    if train_dataset is None:
        logger.error("❌ Failed to create TensorFlow datasets. Exiting.")
        sys.exit(1)

    # Determine input shape and number of classes for the model
    # These should now be directly from the ml_processor.process_ml_data call
    # input_shape is already set above
    # num_classes is already set above

    if input_shape is None: # Redundant check but good for safety
        logger.error("❌ Could not determine input_shape from training dataset. Exiting.")
        sys.exit(1)
    
    logger.info(f"Inferred input_shape for model: {input_shape}")
    logger.info(f"Inferred number of output classes/dimensions: {num_classes}")


    # 4. Initialize and Run the Tuner Selector (Chief)
    # The Chief's CMdtunerSelector will create and manage the Oracle
    tuner_config = CMdtunerSelector(
        backend=tune_params.get('backend', 'tensorflow'),
        tuner_id=TUNER_ID_CHIEF,
        project_name=PROJECT_PATH.name, # Re-added 'project_name' argument
        log_dir=str(LOGDIR), # Pass as string
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset, # Pass test_dataset for final evaluation
        input_shape=input_shape,
        num_classes=num_classes, # Use the dynamically determined num_classes
        max_trials=NUM_TRIALS_CHIEF, # Chief runs all trials
        overwrite=OVERWRITE, # Pass the 'overwrite' argument
        hypermodel_params=all_params, # Pass all_params to the tuner for configuration
        is_chief=True, # Mark as chief
        oracle_url=oracle_url, # Pass Oracle URL
        oracle_directory=str(oracle_full_path) # Pass Oracle directory
    )

    logger.info(f"Chief {TUNER_ID_CHIEF} starting its tuning process...")
    tuner_config.run() # Call the 'run' method for chief

    best_model = tuner_config.get_best_model()

    if best_model:
        logger.info("Best model found. Proceeding with evaluation and saving.")
        # Evaluate the best model
        
        # Check the backend to pass the correct test data format
        if tuner_config.backend == "tensorflow":
            if test_dataset is not None:
                logger.info(f"TensorFlow: Evaluating with test_dataset.")
                # Pass the TensorFlow Dataset directly
                eval_results = tuner_config.evaluate_model(best_model, test_dataset)
                logger.info(f"Final evaluation results: {eval_results}")
            else:
                logger.warning("No TensorFlow test dataset available for final evaluation.")
                eval_results = {}
        elif tuner_config.backend == "pytorch":
            if X_test.size > 0 and y_test.size > 0: # Check if the numpy arrays are not empty
                logger.info(f"PyTorch: Test data extracted. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                # Pass X_test, y_test as a tuple (or separate args) for PyTorch as it expects numpy/tensors
                eval_results = tuner_config.evaluate_model(best_model, (X_test, y_test)) # Assuming evaluate_model takes tuple
                logger.info(f"Final evaluation results: {eval_results}")
            else:
                logger.warning("No PyTorch test data available for final evaluation.")
                eval_results = {}
        else:
            logger.warning(f"Evaluation not supported for backend: {tuner_config.backend}. Skipping final evaluation.")
            eval_results = {}


        # Save the best model
        model_save_path = PROJECT_PATH / "best_model.h5"
        try:
            best_model.save(model_save_path)
            logger.info(f"✅ Best model saved to: {model_save_path}")

            # Optionally convert to ONNX
            try:
                # Ensure input_signature matches what the model expects
                # The batch_size dimension needs to be None for ONNX conversion
                # input_shape from ml_processor is (sequence_length, num_features)
                # So the full input_signature should be (None, sequence_length, num_features)
                input_signature_for_onnx = [tf.TensorSpec([None, *input_shape], dtype=tf.float32)]
                
                onnx_model_path = PROJECT_PATH / "best_model.onnx"
                model_proto, _ = tf2onnx.convert.from_keras(best_model, input_signature=input_signature_for_onnx, opset=13)
                with open(onnx_model_path, "wb") as f:
                    f.write(model_proto.SerializeToString())
                logger.info(f"✅ Model successfully converted to ONNX and saved at {onnx_model_path}")

                # Verify ONNX model with onnx checker
                onnx_model = onnx.load(onnx_model_path)
                checker.check_model(onnx_model)
                logger.info("✅ ONNX model check passed.")

                # Test ONNX inference with onnxruntime
                ort_session = ort.InferenceSession(str(onnx_model_path)) # Convert Path to string
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name

                # Use a small subset of X_test for ONNX inference test
                # Ensure the test_input has the correct batch dimension (None, timesteps, features)
                test_input = X_test[:1].astype(np.float32) # Get first sample, ensure float32
                if test_input.ndim == 2: # If input is (timesteps, features) without batch, add batch dim
                    test_input = np.expand_dims(test_input, axis=0)

                ort_outs = ort_session.run([output_name], {input_name: test_input})
                logger.info(f"✅ ONNX Runtime inference test successful. Output shape: {ort_outs[0].shape}")

            except ImportError:
                logger.warning("tf2onnx, onnx, or onnxruntime not installed. Skipping ONNX conversion/verification.")
            except Exception as e:
                logger.error(f"❌ Failed to convert or verify ONNX model: {e}", exc_info=True)
            finally:
                pass
        except Exception as e:
            logger.error(f"❌ Failed to save model or during ONNX process: {e}", exc_info=True)
    else:
        logger.info("Skipping final evaluation and model saving as no best model was found.")

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
        sys.exit(1)
        
    try:
        main()
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")