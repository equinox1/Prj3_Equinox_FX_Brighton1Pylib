#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                            tsNeuroPredictWinMql_worker.py        |
# |                        Refactored with CMdtunerSelector          |
# +------------------------------------------------------------------+

import os
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path # Ensure Path is imported for type hints and path operations
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tabulate import tabulate # Added for potential debug logging of DataFrames
import sys # Import the sys module to use sys.exit()

import MetaTrader5 as mt5

# Setup modules
from tsMqlSetup import CMqlSetup
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides
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

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__,
    loglevel='INFO',
    logfile='tsneuropredict_app.log',
    backend=backend_for_log
)

# Retrieve global log file and directory paths from environment variables.
# These variables are expected to be set by the multiworker_launcher.
LOGDIR = base_params.get('mp_glob_base_log_path', Path('Logdir'))
if isinstance(LOGDIR, str):
    LOGDIR = Path(LOGDIR)
LOGFILE = app_params.get('LOGFILE', 'tsneuropredict_app.log')


# Global tuner configuration
TUNER_ID = os.environ.get('TUNER_ID', 'default_worker')
ORACLE_DIR = LOGDIR / "tsOracle" # Oracle working directory
MODEL_DIR = base_params.get('mp_glob_base_path', Path.cwd()) / base_params.get('mp_glob_sub_ml_src_modeldata', 'tsModelData') # Path to save models
MODEL_NAME = tune_params.get('ml_model_name', 'tsneuromodel')


# Set TensorFlow mixed precision policy
policy = mixed_precision.Policy(app_params.get('precision', 'float32'))
mixed_precision.set_global_policy(policy)
logger.info(f"TensorFlow global mixed precision policy set to: {policy.name}")


def build_and_compile_model(hp, input_shape):
    """
    Builds a deep learning model for Keras Tuner.
    
    Args:
        hp: HyperParameters object from KerasTuner.
        input_shape (tuple): The shape of the input data (timesteps, features).
    
    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            hp.Int('units_1', min_value=32, max_value=256, step=32),
            return_sequences=True,
            input_shape=input_shape
        ),
        tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.LSTM(
            hp.Int('units_2', min_value=32, max_value=256, step=32)
        ),
        tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(1) # Output layer for regression (e.g., predicting next price change)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mse', # Mean Squared Error for regression
        metrics=['mae'] # Mean Absolute Error as an additional metric
    )
    return model


def main():
    logger.info(f"Worker {TUNER_ID} main function started.")
    
    # 1. Initialize OracleClient for communication with OracleServer
    oracle_client = OracleClient() # Initialize without tuner_id
    # Register the client with the OracleServer
    registration_response = oracle_client.register_client(tuner_id=TUNER_ID)
    if registration_response:
        logger.info(f"OracleClient initialized and registered for tuner_id: {TUNER_ID}")
    else:
        logger.error(f"Failed to register OracleClient for tuner_id: {TUNER_ID}. Exiting.")
        sys.exit(1)

    # 2. Data Loading (using CDataLoader and CDataProcess)
    # 2. Data Loading (using CDataLoader and CDataProcess)
    try:
        # Extract explicit positional arguments for CDataLoader from app_params
        primary_symbol = app_params.get('mp_app_primary_symbol', 'EURUSD')
        timeframe_str = app_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_H4') # Sourced from app_params
        start_date = app_params.get('mp_app_start_date', '2023-01-01')
        end_date = app_params.get('mp_app_end_date', datetime.now().strftime('%Y-%m-%d'))
        data_path = base_params.get('mp_glob_base_data_path', 'Mql5Data') # Sourced from base_params for consistency with chief

        # Dynamically resolve timeframe string to mt5 constant
        try:
            timeframe = getattr(mt5, timeframe_str.split('.')[-1])
            logger.info(f"Resolved timeframe: {timeframe_str} to MT5 constant {timeframe}")
        except AttributeError:
            logger.error(f"Invalid timeframe string: {timeframe_str}. Falling back to mt5.TIMEFRAME_H4.")
            timeframe = mt5.TIMEFRAME_H4 # Fallback
        
        # Collect remaining keyword arguments for CDataLoader
        # These are parameters expected by CDataLoader's __init__ method via **kwargs
        data_loader_kwargs = {
            'mp_data_rows': app_params.get('mp_data_rows', 1000),
            'mp_data_rowcount': app_params.get('mp_data_rowcount', 10000),
            'mp_data_loadapiticks': all_params.get('data', {}).get('mp_data_loadapiticks', True),
            'mp_data_loadapirates': all_params.get('data', {}).get('mp_data_loadapirates', True),
            'mp_data_loadfileticks': all_params.get('data', {}).get('mp_data_loadfileticks', True),
            'mp_data_loadfilerates': all_params.get('data', {}).get('mp_data_loadfilerates', True)
        }

        # Initialize CDataLoader with positional arguments and then keyword arguments
        data_loader = CDataLoader(
            symbol=primary_symbol,
            timeframe=timeframe,
            start_date_str=start_date,
            end_date_str=end_date,
            data_path=data_path,
            **data_loader_kwargs
        )
        logger.info("CDataLoader initialized.")

        # The subsequent data loading logic needs to use data_loader methods.
        # Ensure 'mp_app_cfg_usedata' is used to select the correct loading method.
        used_data_key = app_params.get('mp_app_cfg_usedata', 'df_file_rates') 

        if used_data_key == 'df_file_rates':
            data_df = data_loader.load_data(df_name="df_file_rates") # Assuming load_data can take df_name or similar
        elif used_data_key == 'df_api_rates':
            data_df = data_loader.load_api_rates()
        elif used_data_key == 'df_file_ticks':
            data_df = data_loader.load_file_ticks()
        elif used_data_key == 'df_api_ticks':
            data_df = data_loader.load_api_ticks()
        else:
            logger.error(f"Unsupported mp_app_cfg_usedata: {used_data_key}")
            sys.exit(1)

        if data_df.empty:
            logger.error("Loaded DataFrame is empty. Exiting.")
            sys.exit(1)
        logger.info(f"Data loaded successfully. Initial shape: {data_df.shape}")
        
        # Process data using CDataProcess
        # Ensure CDataProcess also has access to all_params if it needs it internally
        data_processor = CDataProcess(
            df=data_df,
            all_params=all_params, # Pass all_params to CDataProcess
            project_dir=None # Or pass an appropriate project_dir if needed by CDataProcess
        )
        processed_df = data_processor.process_data()

        if processed_df.empty:
            logger.error("Processed DataFrame is empty after CDataProcess. Exiting.")
            sys.exit(1)
        logger.info(f"Data processed successfully. Final shape: {processed_df.shape}")
        logger.info("\nProcessed DataFrame Head:\n%s", tabulate(processed_df.head(), headers='keys', tablefmt='psql'))


    except Exception as e:
        logger.error(f"Error during data loading or processing: {e}", exc_info=True)
        sys.exit(1)
    # 3. Prepare data for ML processing (CDMLProcess)
    # Ensure 'mp_ml_input_keyfeat' and 'mp_ml_input_label' are correctly mapped
    # The CDataProcess class should have set these up based on config.
    key_feature = data_processor.mp_ml_input_keyfeat # e.g., 'Close'
    label_feature = data_processor.mp_ml_input_label # e.g., 'Label'
    history_size = data_processor.data_params.get('mp_data_history_size', 5)


    # Initialize CDMLProcess
    ml_processor = CDMLProcess(
        df=processed_df, 
        input_key_feature=key_feature, # Use the actual column name for input feature
        label_key_feature=label_feature, # Use the actual column name for label
        history_size=history_size
    )
    logger.info("CDMLProcess initialized.")
    
    # Generate datasets (X and y)
    X, y = ml_processor.create_datasets()
    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error("X or y dataset is empty after create_datasets. Exiting.")
        sys.exit(1)
    logger.info(f"Datasets created. X shape: {X.shape}, y shape: {y.shape}")

    # Split and prepare TensorFlow datasets
    train_dataset, val_dataset, test_dataset = ml_processor.prepare_tensorflow_datasets(X, y)
    logger.info("TensorFlow datasets prepared.")

    # 4. Distributed Tuning Loop
    tuner_epochs = tune_params.get('tunemodeepochs', 10)
    max_trials_per_worker = tune_params.get('max_trials_per_worker', 100) # Define a limit for trials per worker

    for trial_num in range(max_trials_per_worker):
        logger.info(f"Worker {TUNER_ID}: Requesting trial {trial_num + 1}/{max_trials_per_worker} from OracleServer.")
        
        trial_response = oracle_client.get_trial()
        
        if trial_response is None:
            logger.error("Failed to get trial from Oracle Server. Exiting worker.")
            break
        
        trial_data = trial_response.get("trial")
        if trial_data is None:
            logger.info("Oracle Server reported no more trials available or an empty trial. Shutting down worker.")
            break

        trial_id = trial_data["trial_id"]
        hyperparameters_json = trial_data["hyperparameters"]

        # Reconstruct HyperParameters object from JSON (if necessary, for KerasTuner's build_model signature)
        # For simple cases, you might just extract values directly.
        # This part depends on how 'build_and_compile_model' expects 'hp'
        hp = tf.keras.src.applications.resnet.HyperParameters() # Using a dummy instance, or load properly
        for param_name, param_value in hyperparameters_json.items():
            # This is a simplification; a proper KerasTuner HP object might need more complex reconstruction
            # For 'Int' and 'Float', direct assignment might be sufficient for basic usage.
            if 'units' in param_name:
                hp.Int(param_name, min_value=32, max_value=256, step=32, default=param_value)
            elif 'dropout' in param_name:
                hp.Float(param_name, min_value=0.2, max_value=0.5, step=0.1, default=param_value)
            elif 'learning_rate' in param_name:
                hp.Choice(param_name, values=[1e-2, 1e-3, 1e-4], default=param_value)
            # Add other hyperparameter types as needed

        logger.info(f"Worker {TUNER_ID}: Starting trial {trial_id} with hyperparameters: {hyperparameters_json}")
        
        try:
            # Build the model using the hyperparameters for this trial
            # Ensure input_shape is correct for your data (e.g., (history_size, num_features))
            input_shape = (X.shape[1], X.shape[2]) if X.ndim == 3 else (X.shape[1],)
            model = build_and_compile_model(hp, input_shape)

            # Train the model
            history = model.fit(
                train_dataset,
                epochs=tuner_epochs,
                validation_data=val_dataset,
                verbose=1 # Show progress
            )

            # Get the best validation loss from this trial's training history
            val_loss = min(history.history['val_loss'])
            
            # Report result to OracleServer
            oracle_client.report_trial_result(trial_id, {"val_loss": val_loss})
            oracle_client.update_trial_status(trial_id, status="COMPLETED")
            logger.info(f"Worker {TUNER_ID}: Trial {trial_id} completed with val_loss: {val_loss}")

        except Exception as e:
            logger.error(f"Worker {TUNER_ID}: Error during trial {trial_id} training: {e}", exc_info=True)
            oracle_client.update_trial_status(trial_id, status="FAILED")
            continue # Continue to next trial if current one fails

    logger.info(f"🏁 tsNeuroPredictWinMql_worker.py finished for tuner_id: {TUNER_ID}.")


if __name__ == "__main__":
    # Ensure MetaTrader5 is initialized and finalized with authentication details
    #mt5_login = app_params.get('mp_app_login', 123456) # Replace with your actual login
    #mt5_password = app_params.get('mp_app_password', 'your_password') # Replace with your actual password
    # Default to a common demo server if not specified or empty
    #mt5_server = app_params.get('mp_app_server', 'MetaQuotes-Demo') 

    # Add checks for placeholder credentials
    #if mt5_login == 123456:
    #    logger.warning("⚠️ Using placeholder MetaTrader5 login. Please update 'mp_app_login' in your config.")
    #if mt5_password == 'your_password':
    #    logger.warning("⚠️ Using placeholder MetaTrader5 password. Please update 'mp_app_password' in your config.")
    #if mt5_server == 'your_server_name' or mt5_server == '':
    #    logger.warning("⚠️ Using placeholder/empty MetaTrader5 server. Please update 'mp_app_server' in your config. Defaulting to 'MetaQuotes-Demo'.")


    # ----- Broker Login -----
    logger.info("PARAM HEADER: MP_APP_BROKER: %s", app_params.get('mp_app_broker'))
    broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
    mqqlobj = broker_config.run_mql_login()
    if mqqlobj is True:
        logger.info("Successfully logged in to MetaTrader 5.")
    else:
        logger.info("Failed to login. Error code: %s", mqqlobj)
        sys.exit(1)


    try:
        # Run the main function
        main()
    finally:
        mt5.shutdown()
        logger.info("✅ MetaTrader5 shutdown.")
