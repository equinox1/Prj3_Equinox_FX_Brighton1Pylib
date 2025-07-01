TUNER_ID_CHIEF = "chief" # This constant remains for reference, but worker uses its own ID
#!/usr/bin/env python3
# +------------------------------------------------------------------+\
# |                                    tsNeuroPredictWinMql_worker.py|\
# |                                                    Tony Shepherd |\
# |                                    https://www.xercescloud.co.uk |\
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

logger = logging.getLogger(__name__)
# Onnx and tf2onnx imports (kept for imports, but conversion logic removed from worker main)
try:
    import tf2onnx
    import onnx
    from onnx import checker
    import onnxruntime as ort
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    print("tf2onnx, onnx, or onnxruntime not installed. ONNX conversion/verification will be skipped.")

import MetaTrader5 as mt5

# Import mixed_precision for TensorFlow policy
from tensorflow.keras import mixed_precision

# Custom modules
from tsMqlSetup import CMqlSetup
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
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
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

# Determine global backend (should be passed from launcher)
global_backend = os.environ.get("BACKEND", tune_params.get("backend", "tensorflow"))
os.environ["KERAS_BACKEND"] = global_backend



# Initialize CMqlSetup to get configuration, including precision
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel=app_params.get('LOGLEVEL', 'INFO'),
    warn='ignore',
    precision=app_params.get('TF_PRECISION', 'mixed_float16'),
    tfdebug=app_params.get('TFDEBUG', False),
    num_cores=_estimated_physical_cores,
    num_threads=_logical_cores
)

# Set mixed precision policy if not already set globally by CMqlSetup
policy = mixed_precision.Policy(setup_config.precision)
mixed_precision.set_global_policy(policy)
logger.info(f"✨ Global mixed precision policy set to: {mixed_precision.global_policy().compute_dtype}")

# Oracle Server configuration (worker needs this to connect)
xerces_server = app_params.get('xerces_server', '127.0.0.1')
xerces_port = app_params.get('xerces_port', 9000)
oracle_url = f"http://{xerces_server}:{xerces_port}"
LOGDIR = app_params.get('LOGDIR', 'Logdir')
LOGDIR = Path(LOGDIR)

# --- Main Logic for Worker ---
def main():
    # Get TUNER_ID from environment variable set by the launcher
    tuner_id = os.environ.get('TUNER_ID', 'worker_default')
    logger.info(f"Worker {tuner_id} started. Kicking off data loading and processing.")

    # Model and Tuner Configuration (worker also needs these for building models)
    MODEL_NAME = tune_params.get('ml_model_name', 'tsneuromodel')
    MODEL_DIR = Path(base_params.get('mp_glob_sub_ml_src_modeldata', 'tsModelData'))
    PROJECT_PATH = MODEL_DIR / MODEL_NAME
    PROJECT_PATH.mkdir(parents=True, exist_ok=True) # Ensure project directory exists

    ml_project_id = os.environ.get('mp_glob_sub_ml_baseuniq', str(base_params.get("mp_glob_sub_ml_baseuniq", 777)))
    project_name = f"{MODEL_NAME}_{ml_project_id}"

    # 1. Data Loading
    primary_symbol = app_params.get('mp_app_primary_symbol', 'EURUSD')
    timeframe_str = app_params.get('mp_app_timeframe', 'mt5.TIMEFRAME_H4')
    
    try:
        timeframe = getattr(mt5, timeframe_str.split('.')[-1])
        logger.info(f"Resolved timeframe: {timeframe_str} to MT5 constant {timeframe}")
    except AttributeError:
        logger.error(f"Invalid timeframe string: {timeframe_str}. Falling back to mt5.TIMEFRAME_H4.")
        timeframe = mt5.TIMEFRAME_H4

    start_date = app_params.get('mp_app_start_date', '2023-01-01')
    end_date = app_params.get('mp_app_end_date', datetime.now().strftime('%Y-%m-%d'))
    data_path = base_params.get('mp_glob_base_data_path', 'Mql5Data')

    data_loader_kwargs = {
        'mp_data_rows': all_params.get('data', {}).get('mp_data_rows', 1000),
        'mp_data_rowcount': all_params.get('data', {}).get('mp_data_rowcount', 10000),
        'mp_data_loadapiticks': all_params.get('data', {}).get('mp_data_loadapiticks', True),
        'mp_data_loadapirates': all_params.get('data', {}).get('mp_data_loadapirates', True),
        'mp_data_loadfileticks': all_params.get('data', {}).get('mp_data_loadfileticks', True),
        'mp_data_loadfilerates': all_params.get('data', {}).get('mp_data_loadfilerates', True)
    }

    data_loader = CDataLoader(
        symbol=primary_symbol,
        timeframe=timeframe,
        start_date_str=start_date,
        end_date_str=end_date,
        data_path=data_path,
        **data_loader_kwargs
    )

    all_dfs = data_loader.run_dataloader_services() 
    logger.info(f"Data Loader services finished. Loaded DataFrames: {all_dfs.keys()}")

    used_data_key = app_params.get('mp_app_cfg_usedata', 'df_file_rates')
    main_df = all_dfs.get(used_data_key)

    if main_df is None or main_df.empty:
        logger.error(f"❌ Main DataFrame '{used_data_key}' is not available or is empty after data loading. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded DataFrame '{used_data_key}' with shape: {main_df.shape}")

    # 2. Data Processing
    data_processor = CDataProcess(
        df=main_df,
        all_params=all_params,
        project_dir=PROJECT_PATH
    )
    processed_df = data_processor.process_data()

    if processed_df is None or processed_df.empty:
        logger.error("❌ Processed DataFrame is empty after CDataProcess. Exiting.")
        sys.exit(1)

    logger.info(f"Processed DataFrame shape after CDataProcess: {processed_df.shape}")

    # 3. Machine Learning Data Preparation
    ml_processor = CDMLProcess(df=processed_df, all_params=all_params, project_dir=PROJECT_PATH)
    X_final, y_final, input_shape, num_classes = ml_processor.process_ml_data()

    if X_final is None or X_final.size == 0 or y_final is None or y_final.size == 0:
        logger.error("❌ Processed X or y are empty after ML processing. Exiting.")
        sys.exit(1)
    
    logger.info(f"ML Processed X shape: {X_final.shape}, Y shape: {y_final.shape}")

    random_state = tune_params.get('seed', 42)
    X_train, X_val, X_test, y_train, y_val, y_test = ml_processor.split_dataset(
        X_final, y_final,
        train_size=tune_params.get('train_split', 0.7),
        val_size=tune_params.get('val_split', 0.15),
        test_size=tune_params.get('test_split', 0.15),
        random_state=random_state
    )
    logger.info(f"Data split into: Train X: {X_train.shape}, Y: {y_train.shape} | Val X: {X_val.shape}, Y: {y_val.shape} | Test X: {X_test.shape}, Y: {y_test.shape}")

    train_dataset, val_dataset, test_dataset = ml_processor.create_tf_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=tune_params.get('batch_size', 32),
        shuffle_buffer=tune_params.get('buffer_size', 1000)
    )

    if train_dataset is None:
        logger.error("❌ Failed to create TensorFlow datasets. Exiting.")
        sys.exit(1)

    if input_shape is None:
        logger.error("❌ Could not determine input_shape from training dataset. Exiting.")
        sys.exit(1)
    
    logger.info(f"Inferred input_shape for model: {input_shape}")
    logger.info(f"Inferred number of output classes/dimensions: {num_classes}")

    # 4. Initialize and Run the Tuner Selector (Worker)
    # The worker connects to the existing OracleServer
    tuner_selector = CMdtunerSelector(
        backend=global_backend,
        tuner_id=tuner_id, # Use the dynamic tuner_id for the worker
        project_name=project_name, # Workers also need project_name for local files/dirs
        log_dir=str(LOGDIR),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_shape=input_shape,
        num_classes=num_classes,
        max_trials=tune_params.get('num_trials', 50), # Max trials is less relevant for worker, but keep for consistency
        overwrite=False, # Workers should not overwrite the Oracle state
        hypermodel_params=all_params,
        is_chief=False, # THIS IS THE CRUCIAL CHANGE FOR A WORKER
        oracle_url=oracle_url, # Worker MUST connect to a remote Oracle
        oracle_directory=str(LOGDIR / "oracle_server") # Worker needs this path for its local KerasTuner files
    )

    logger.info(f"Worker {tuner_id} starting its trial execution loop...")
    tuner_selector.run() # This method now contains the worker's trial fetching loop
    logger.info(f"Worker {tuner_id} trial execution completed.")

    # Workers do not save the best model or perform ONNX conversion.
    # That is the responsibility of the chief process.

    logger.info("🏁 tsNeuroPredictWinMql_worker.py finished.")


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
