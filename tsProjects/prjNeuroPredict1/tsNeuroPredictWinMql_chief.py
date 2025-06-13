#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                    tsNeuroPredictWinMql_chief.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+

<<<<<<< HEAD
import os
import sys
import logging # Import logging, but do NOT configure the root logger here.
=======

import os
import logging
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
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
<<<<<<< HEAD
from tsMqlSetup import CMqlSetup # Import CMqlSetup for non-logging config, but not for root logger setup.
from tsMqlOverrides import CMqlOverrides
=======

>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr

from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlMLProcess import CDMLProcess
<<<<<<< HEAD

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
=======
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
from tsMqlMLTuner.tsMqlMLTunerModTorch import PyTorchTuner


os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNER_ID"] = "chief"
tuner_id = os.environ.get("TUNER_ID", "chief")

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides

env_backend = os.environ.get("MLTUNE_BACKEND", "tensorflow")
env_gtuner = os.environ.get("GTUNER_MODEL", env_backend)

mql_overrides = CMqlOverrides()
mql_overrides.env.override_params({
    "mltune": {"backend": env_backend},
    "app": {"gtuner_model": env_gtuner}
})

app_params = mql_overrides.env.all_params().get("app", {})
gtuner_model = app_params.get('gtuner_model', 'pytorch')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
<<<<<<< HEAD
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
=======
    num_cores=8,
    num_threads=1
)

global_logdir, global_logfile = setup_config.set_log_dir(
    logdir=None,
    logfile=xerces_logfile,
    servername=xerces_servername,
    ltuner=gtuner_model
)

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)


# strategy setup
strategy = setup_config.get_computation_strategy()
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

diststrategy = 'tf.distribute.MultiWorkerMirroredStrategy'
logger.info(f"Distribution strategy: {diststrategy}")
gtuner_type = 'distributed'
gtuner_mode = 'random'
gmodscale = 8
gstandalone = False


# ----- Main Function -----
def main(logger):
    #with strategy.scope():
        # ---- Configuration ----
        is_chief = tuner_id.lower() == "chief"
        # Setup environment and retrieve parameters
        print("Start Main Setting up environment...")
        utils_config = CUtilities()
        mql_overrides = CMqlOverrides()  # Uses defaults if no config.yaml provided

        base_params = mql_overrides.env.all_params().get("base", {})
        data_params = mql_overrides.env.all_params().get("data", {})
        feat_params = mql_overrides.env.all_params().get("features", {})
        ml_params = mql_overrides.env.all_params().get("ml", {})
        mltune_params = mql_overrides.env.all_params().get("mltune", {})
        app_params = mql_overrides.env.all_params().get("app", {})
        
        # Log the logfile location; ensure logdir is not None.
        logdir = base_params.get('mp_glob_base_log_path') or global_logdir
        logdir = global_logdir if logdir is None else logdir
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, 'tsneuropredict_app.log')
        logger.info(f"Logfile: {logfile}")
        
         # ----- Model Tuning and Setup -----
        mql_overrides.env.override_params({"app": {'mp_app_ml_hard_run': False}})
        mql_overrides.env.override_params({"mltune": {'batch_size': 8}})
        mql_overrides.env.override_params({"data": {'mp_data_timeframe': mt5.TIMEFRAME_H4}})
        logger.info("Main: mp_app_ml_hard_run: %s", app_params.get('mp_app_ml_hard_run', True))
        logger.info("Main: mp_ml_mbase_path: %s", base_params.get('mp_glob_base_ml_project_dir', None))
        logger.info("Main: batch_size: %s", base_params.get('batch_size', None))
        logger.info("Main: mp_data_timeframe: %s", data_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_H4'))

        # Scale the model
        modscale = gmodscale
        logger.info("Main: Model Scale: %s", modscale)
    
        # ----- Load Reference class and time variables -----
        lp_timeframe_name = data_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_H4')
        logger.info("Main:Chief Timeframe Name: %s", lp_timeframe_name)
      
        reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
        
        # Adjust TIME_CONSTANTS handling in case it's a list.
        time_constants = reference_config.TIME_CONSTANTS
        if isinstance(time_constants, list):
            time_constants = time_constants[0]

        # Extract time constants
        UNIT = time_constants["UNIT"]["SECOND"]
        MINUTE = reference_config.get_timevalue('MINUTE')
        HOUR = reference_config.get_timevalue('HOUR')
        DAY = reference_config.get_timevalue('DAY')
        CURRENT_TIME = reference_config.get_current_time()
        CURRENTDAY = CURRENT_TIME["CURRENTDAY"]
        CURRENTMONTH = CURRENT_TIME["CURRENTMONTH"]
        CURRENTYEAR = CURRENT_TIME["CURRENTYEAR"]
        TIMEZONE = CURRENT_TIME["TIMEZONE"]
        TIMEFRAME = CURRENT_TIME["TIMEFRAME"]
        timeval = HOUR  # used for window creation
        logger.info(f"Timezone: {TIMEZONE}")
        logger.info(f"Timeframe: {TIMEFRAME}")

        mql_overrides.env.override_params({"data": {"mp_data_rows": 1000}})
        mql_overrides.env.override_params({"data": {"mp_data_rowcount": 100000}})
       
        rows = data_params.get('mp_data_rows', 1000)
        rowcount = data_params.get('mp_data_rowcount', 10000)
        logger.info(f"Timeframe Name: {lp_timeframe_name}, Rows: {rows}, Rowcount: {rowcount}")

        # ----- Broker Login -----
        logger.info("PARAM HEADER: MP_APP_BROKER: %s", app_params.get('mp_app_broker'))
        broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
        mqqlobj = broker_config.run_mql_login()
        if mqqlobj is True:
            logger.info("Successfully logged in to MetaTrader 5.")
        else:
            logger.info("Failed to login. Error code: %s", mqqlobj)

        # ----- Data Loader and Process Initialization -----
        data_loader_config = CDataLoader()
        data_process_config = CDataProcess(mp_unit=UNIT)
        ml_process_config = CDMLProcess()

        # ----- Data Loading and Processing -----
        mp_data_history_size = data_params.get('mp_data_history_size', 1)
        mv_data_utc_from = data_loader_config.set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        mv_data_utc_to = data_loader_config.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        logger.info(f"Main: UTC From: {mv_data_utc_from}")
        logger.info(f"Main: UTC To: {mv_data_utc_to}")

        data_loader_config = CDataLoader(
            lp_utc_from=mv_data_utc_from,
            lp_utc_to=mv_data_utc_to,
            lp_timeframe=lp_timeframe_name,
            lp_app_primary_symbol=app_params.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD')),
            lp_app_rows=rows,
            lp_app_rowcount=rowcount
        )
        df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = data_loader_config.run_dataloader_services()
        logger.info("Loaded: Data API Ticks: %s, Data API Rates: %s, Data File Ticks: %s, Data File Rates: %s",
                    df_api_ticks.shape, df_api_rates.shape, df_file_ticks.shape, df_file_rates.shape)

        df_api_ticks = data_process_config.run_dataprocess_services(df=df_api_ticks, df_name='df_api_ticks')
        df_api_rates = data_process_config.run_dataprocess_services(df=df_api_rates, df_name='df_api_rates')
        df_file_ticks = data_process_config.run_dataprocess_services(df=df_file_ticks, df_name='df_file_ticks')
        df_file_rates = data_process_config.run_dataprocess_services(df=df_file_rates, df_name='df_file_rates')
        utils_config.run_mql_print(df=df_api_ticks, df_name='df_api_ticks', hrows=5, colwidth=30, app='data processing')
        utils_config.run_mql_print(df=df_api_rates, df_name='df_api_rates', hrows=5, colwidth=30, app='data processing')
        utils_config.run_mql_print(df=df_file_ticks, df_name='df_file_ticks', hrows=5, colwidth=30, app='data processing')
        utils_config.run_mql_print(df=df_file_rates, df_name='df_file_rates', hrows=5, colwidth=30, app='data processing')
        datafile = df_file_rates

        # ----- Add Time Index to Data -----
        column_features = datafile.columns[1:]
        datafile = datafile[[datafile.columns[0]] + list(column_features)]
        utils_config.run_mql_print(df=datafile, df_name='df_file_rates', hrows=5, colwidth=30, app='datafile')
        logger.info("DataFrame index: %s", datafile.index)

        # ----- Create Window Parameters -----
        logger.info("Creating the 24 hour prediction window with timeval: %s and HOUR: %s", timeval, HOUR)
        back_window, forward_window, pred_width = ml_process_config.create_ml_window(timeval=HOUR)
        total_window_size = back_window + forward_window
        logger.info("Create Window: Back Window: %s, Forward Window: %s, Prediction Width: %s", back_window, forward_window, pred_width)

        mql_overrides.env.override_params({"mltune": {"total_window_size": total_window_size}})
        mql_overrides.env.override_params({"mltune": {"input_width": back_window}})
        mql_overrides.env.override_params({"mltune": {"label_width": forward_window}})
        mql_overrides.env.override_params({"mltune": {"shift": pred_width}})
        mql_overrides.env.override_params({"mltune": {'tf_param_epochs': 1}})
        mql_overrides.env.override_params({"mltune": {'distribution_strategy': diststrategy}})
        mql_overrides.env.override_params({"mltune": {'tunertype': gtuner_type}})
        mql_overrides.env.override_params({"mltune": {'tunemode': gtuner_mode}})
        
       
        mltune_overrides = mql_overrides.env.all_params().get("mltune", {})
        logger.info("OverRidden: ML Tuning Parameters: %s", mltune_overrides)
        logger.info("OverRidden: Total Window Size: %s", mltune_overrides.get("total_window_size", total_window_size))
        logger.info("OverRidden: Input Width: %s", mltune_overrides.get("Input Width", back_window))
        logger.info("OverRidden: Label Width: %s", mltune_overrides.get("Label Width", forward_window))
        logger.info("OverRidden: Shift: %s", mltune_overrides.get("Shift", pred_width))
        logger.info("OverRidden: Distribution Strategy: %s", mltune_overrides.get("distribution_strategy", diststrategy))
        logger.info("OverRidden: Tuner Type: %s", mltune_overrides.get("tunertype", gtuner_type))
        logger.info("OverRidden: Tuner Mode: %s", mltune_overrides.get("tunemode", gtuner_mode))
        
          # ----- Select Features and Labels -----
        features = ml_params.get("mp_ml_input_keyfeat", "Close")
        features_scaled = ml_params.get("mp_ml_input_keyfeat_scaled", "Close_Scaled")
        label1 = ml_params.get("mp_ml_output_label", "Label")
        logger.info("Main: Features: %s, Features Scaled: %s, Label1: %s", features, features_scaled, label1)

        # ----- Generate X and y -----
        datafile_X, datafile_y = ml_process_config.Create_Xy_input_and_target(
            datafile, back_window=back_window, forward_window=forward_window, features=[features]
        )
        logger.info("Input shape: %s, Target shape: %s", datafile_X.shape, datafile_y.shape)

        # ----- Scaling the Input Features -----
        # Reshape X from (n_samples, back_window, n_features) to 2D for scaling
        nsamples, nsteps, nfeatures = datafile_X.shape
        X_reshaped = datafile_X.reshape((nsamples * nsteps, nfeatures))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        # Reshape back to the original 3D shape
        datafile_X_scaled = X_scaled.reshape(datafile_X.shape)
        if datafile_X_scaled.ndim == 4:
            datafile_X_scaled = np.squeeze(datafile_X_scaled, axis=-1)
        elif datafile_X_scaled.ndim == 2:
            datafile_X_scaled = np.expand_dims(datafile_X_scaled, axis=-1)

        # Optionally, scale the targets (uncomment if desired)
        y_reshaped = datafile_y.reshape((-1, 1))
        y_scaled = scaler.fit_transform(y_reshaped)
        datafile_y_scaled = y_scaled.reshape(datafile_y.shape)

        # ----- Split Data (using scaled inputs) -----
        seed = mltune_params.get('seed', 42)
        n_samples = len(datafile_X_scaled)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        X_train, X_val, X_test, y_train, y_val, y_test = ml_process_config.manual_split_data(
            datafile_X_scaled, datafile_y, train_end, val_end
        )
        logger.info("Train samples: %s", X_train.shape[0])
        logger.info("Validation samples: %s", X_val.shape[0])
        logger.info("Test samples: %s", X_test.shape[0])

         # ----- Convert to TensorFlow Dataset -----
        batch_size = ml_params.get('batch_size', 1024)
     
        buffer_size = ml_params.get('buffer_size', 10000)
        logger.info("Buffer size: %s", buffer_size)
        train_dataset, val_dataset, test_dataset = ml_process_config.create_simple_tf_dataset(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size, buffer_size=buffer_size
        )
        logger.info("Train dataset: %s", train_dataset)
        logger.info("Validation dataset: %s", val_dataset)
        logger.info("Test dataset: %s", test_dataset)
        # Check the dataset shapes
        logger.info("Train dataset shape: %s", train_dataset.element_spec[0].shape)
        logger.info("Validation dataset shape: %s", val_dataset.element_spec[0].shape)
        logger.info("Test dataset shape: %s", test_dataset.element_spec[0].shape)
        input_shape = train_dataset.element_spec[0].shape[1:]  # ✅ Drop batch dimension (None)

        data_input_shape = input_shape
        output_shape = train_dataset.element_spec[1].shape

        
        mql_overrides.env.override_params({"mltune": {"input_shape": input_shape}})
        mql_overrides.env.override_params({"mltune": {"output_shape": output_shape}})
        mql_overrides.env.override_params({"mltune": {"data_input_shape": input_shape}})
        mltune_overrides = mql_overrides.env.all_params().get("mltune", {})

        logger.info("Input shape: %s", input_shape)
        logger.info("Output shape: %s", output_shape)

        # ----- Model Scale ----- see start of file for model scale
        mql_overrides.env.override_params({"mltune": {'all_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'cnn_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'lstm_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'gru_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'trans_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'transh_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'transff_modelscale': modscale}})
        mql_overrides.env.override_params({"mltune": {'dense_modelscale': modscale}})

        all_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('all_modelscale', 1)
        cnn_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('cnn_modelscale', 1)
        lstm_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('lstm_modelscale', 1)
        gru_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('gru_modelscale', 1)
        trans_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('trans_modelscale', 1)
        transh_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('transh_modelscale', 1)
        transff_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('transff_modelscale', 1)
        dense_modelscale = mql_overrides.env.all_params().get('mltune', {}).get('dense_modelscale', 1)

        # Tune overrides
        mql_overrides.env.override_params({"mltune": {'unitmin': int(32/modscale)}})
        mql_overrides.env.override_params({"mltune": {'unitmax': int(512/modscale)}})
        mql_overrides.env.override_params({"mltune": {'unitstep': int(32/modscale)}})
        mql_overrides.env.override_params({"mltune": {'defaultunits': int(128/modscale)}})
        mql_overrides.env.override_params({"mltune": {'max_epochs': 10}})
        mql_overrides.env.override_params({"mltune": {'min_epochs': 1}})
        mql_overrides.env.override_params({"mltune": {'tunemodeepochs': True}})
        mql_overrides.env.override_params({"mltune": {'tune_new_entries': True}})

        # Misc overrides
        mql_overrides.env.override_params({"mltune": {'mp_ml_show_plot': True}})
        mql_overrides.env.override_params({"mltune": {'ONNX_save': True}})
        mql_overrides.env.override_params({"mltune": {'overwrite': False}})
        mql_overrides.env.override_params({"mltune": {'tuner_id': tuner_id}})
        
        unitmin = mql_overrides.env.all_params().get('mltune', {}).get('unitmin', None)
        unitmax = mql_overrides.env.all_params().get('mltune', {}).get('unitmax', None)
        unitstep = mql_overrides.env.all_params().get('mltune', {}).get('unitstep', None)
        defaultunits = mql_overrides.env.all_params().get('mltune', {}).get('defaultunits', None)

        # Note Epochs is extracted once the model has tuned and found best epoch this a declare of defaults
        epochs = mql_overrides.env.all_params().get('mltune', {}).get('epochs', None)
        tune_new_entries = mql_overrides.env.all_params().get('mltune', {}).get('tune_new_entries', None)

        logger.info("Main: ML Tuning Parameters: %s", unitmin)
        logger.info("Main: ML Tuning Parameters: %s", unitmax)
        logger.info("Main: ML Tuning Parameters: %s", unitstep)
        logger.info("Main: ML Tuning Parameters: %s", defaultunits)
        logger.info("Main: ML Tuning Parameters: %s", epochs)
        logger.info("Main: ML Tuning Parameters: %s", tune_new_entries)

        mp_ml_mbase_path = base_params.get('mp_glob_base_ml_project_dir', None)
        mp_ml_model_name = base_params.get('mp_glob_sub_ml_model_name', None)
        mp_ml_hard_run = app_params.get('mp_app_ml_hard_run', True)
        mp_ml_tf_param_epochs = base_params.get('mp_ml_tf_param_epochs', 1)
        ONNX_save = base_params.get('onnx_save', False)
        mp_glob_sub_ml_src_modeldata = base_params.get('mp_glob_sub_ml_src_modeldata', None)
        mp_symbol_primary = base_params.get('lp_app_primary_symbol', 'EURUSD')

        logger.info("Main Model Check: mp_ml_mbase_path: %s", mp_ml_mbase_path)
        logger.info("Main Model Check: mp_ml_model_name: %s", mp_ml_model_name)
        logger.info("Main Model Check: mp_ml_hard_run: %s", mp_ml_hard_run)
        logger.info("Main Model Check: mp_ml_tf_param_epochs: %s", mp_ml_tf_param_epochs)
        logger.info("Main Model Check: ONNX_save: %s", ONNX_save)
        logger.info("Main Model Check: mp_glob_sub_ml_src_modeldata: %s", mp_glob_sub_ml_src_modeldata)
        logger.info("Main Model Check: mp_symbol_primary: %s", mp_symbol_primary)
        logger.info("Main Model get all_modelscale: %s", mql_overrides.env.all_params().get('mltune', {}).get('all_modelscale', 1))

       # Log parameter details
        logger.info("Main Base Parameters:")
        for key, value in base_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Main Data Parameters:")
        for key, value in data_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Main ML Parameters:")
        for key, value in ml_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Main ML Tuning Parameters:")
        for key, value in mltune_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Main App Parameters:")
        for key, value in app_params.items():
            logger.info(f"  {key}: {value}")

        
        xerces_server = app_params.get('xerces_server', '192.168.1.103')
        xerces_port = app_params.get('xerces_port', 9000)
        oracle = OracleClient(host=xerces_server, port=xerces_port)

        # Conditional Tuner
        tuner_config = CMdtunerSelector(
            oracle=OracleClient(host=xerces_server, port=xerces_port),
            hypermodel_params=mql_overrides.env.all_params(),
            traindataset=train_dataset,
            valdataset=val_dataset,
            testdataset=test_dataset,
            castmode='float16',
        )
            
        # --- Now run tuning normally ---
        runtuner = tuner_config.run_search()
        tuner_config.export_best_model(ftype='tf')

        logger.info("Main Model Check: mp_ml_mbase_path: %s", mp_ml_mbase_path)
        best_model = tuner_config.check_and_load_model(mp_ml_mbase_path, ftype='tf')

        if best_model is None:
            logger.info("No best model loaded. Running tuner search (default run).")
            runtuner = tuner_config.run_search()
            tuner_config.export_best_model(ftype='tf')
        elif mp_ml_hard_run:
            logger.info("Running tuner search (hard run).")
            runtuner = tuner_config.run_search()
            tuner_config.export_best_model(ftype='tf')
        else:
            logger.info("Best model loaded successfully.")
            runtuner = True

        # ----- Train and Evaluate the Model -----
        logger.info("Model: Loading file from directory %s, filename: %s", mp_ml_mbase_path, mp_ml_model_name)
        load_model = tuner_config.check_and_load_model(mp_ml_mbase_path, ftype='tf')
        if load_model is not None:
            best_model = load_model
            logger.info("Model: Best model: %s", best_model.name)
            best_model.summary(print_fn=lambda x: logger.info(x))
            
            # Clear any previous session to free up resources
            tf.keras.backend.clear_session()

            try:
                # Set up callbacks (e.g., early stopping) if desired
                callbacks = [
                    tensorboard_cb,
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                logger.info("Training the best model...")

                logger.info("Best Epochs: %s, tf_epochs: %s", epochs, mp_ml_tf_param_epochs)

                best_model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                )
                logger.info("Training completed.")

                # ----- Model Evaluation -----
                # Predict on the test dataset
                y_pred = best_model.predict(test_dataset)
                # Combine the batched true labels into a single array
                y_test_true = np.concatenate([y for _, y in test_dataset], axis=0)
                
                # Compute evaluation metrics
                mse_value = mean_squared_error(y_test_true, y_pred)
                mae_value = mean_absolute_error(y_test_true, y_pred)
                r2_value = r2_score(y_test_true, y_pred)
                
                logger.info("Evaluation on Test Data:")
                logger.info("MSE: %s", mse_value)
                logger.info("MAE: %s", mae_value)
                logger.info("R2 Score: %s", r2_value)
                
                # ----- Plotting Predictions vs Actuals -----
                plt.figure(figsize=(10, 6))
                plt.plot(y_test_true, label='Actual Price')
                plt.plot(y_pred, label='Predicted Price')
                plt.xlabel("Sample Index")
                plt.ylabel("Price Value")
                plt.title("Price Prediction Evaluation")
                plt.legend()
                # Save the plot to the specified path
                logger.info("PLOT: mp_glob_sub_ml_src_modeldata: %s", mp_glob_sub_ml_src_modeldata)
                logger.info("PLOT: mp_symbol_primary: %s", mp_symbol_primary)

                print("PLOT: mp_symbol_primary: %s", mp_symbol_primary)
                print("PLOT: mp_ml_mbase_path: %s", mp_ml_mbase_path)
                print("PLOT: mp_glob_sub_ml_src_modeldata: %s", mp_glob_sub_ml_src_modeldata)

                plot_path = os.path.join(mp_ml_mbase_path, f"price_prediction_plot_{mp_symbol_primary}.png")
                logger.info("Plot Path: %s", plot_path)
                print("Plot Path: %s", plot_path)
                plt.savefig(plot_path)
                
                # Close the plot to free up memory
                plt.close()
                logger.info("Price prediction plot saved at: %s", plot_path)
                
                # ----- ONNX Model Export -----
                if ONNX_save:
                    try:
                        # Ensure mp_ml_data_type is defined (defaulting to 'data' if not provided)
                        mp_ml_data_type = base_params.get('mp_ml_data_type', 'data')
                        mp_output_path = os.path.join(mp_glob_sub_ml_src_modeldata, f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx")
                        logger.info("Output Path: %s", mp_output_path)
                        opset_version = 17
                        spec = [tf.TensorSpec(best_model.input_shape, tf.float16, name="input")]
                        onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)
                        onnx.save_model(onnx_model, mp_output_path)
                        logger.info("Model saved to %s", mp_output_path)
                        checker.check_model(onnx_model)
                        logger.info("ONNX model is valid. ONNX Runtime version: %s", ort.__version__)
                    except Exception as e:
                        logger.error("ONNX conversion failed: %s", str(e))
            finally:
                mt5.shutdown()
                logger.info("Finished.")
        else:
            logger.info("No data loaded; exiting.")
            mt5.shutdown()
            logger.info("Finished.")
        
if __name__ == "__main__":
    main(logger)
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
