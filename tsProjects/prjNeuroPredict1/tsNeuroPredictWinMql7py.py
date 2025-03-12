#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +------------------------------------------------------------------+

import os
import pathlib
from pathlib import Path
import json
from datetime import datetime, date
import pytz

# Data packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Machine Learning packages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Extra modules needed for ONNX conversion and MetaTrader5 (adjust if not used)
import tf2onnx
import onnx
from onnx import checker
import onnxruntime as ort
import MetaTrader5 as mt5

# Custom modules
from tsMqlSetup import CMqlSetup
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides
from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlMLTune import CMdtuner
from tsMqlMLProcess import CDMLProcess

# ----- Logging configuration -----
logdir = r"C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir"
os.makedirs(logdir, exist_ok=True)
logfile = os.path.join(logdir, 'tsneuropredict_app.log')

import logging
# Remove any existing handlers to ensure our configuration is used.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile, mode='w')


formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s -%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)


logger.info("Logging configured successfully with FileHandler.")

# ----- Setup platform -----
setup_config = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = setup_config.get_computation_strategy()

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# Set package options
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

# ----- Main Function -----
def main(logger):
    with strategy.scope():
        # Setup environment and retrieve parameters
        utils_config = CUtilities()
        mql_overrides = CMqlOverrides()  # Uses defaults if no config.yaml provided

        base_params = mql_overrides.env.all_params().get("base", {})
        data_params = mql_overrides.env.all_params().get("data", {})
        feat_params = mql_overrides.env.all_params().get("feat", {})
        ml_params = mql_overrides.env.all_params().get("ml", {})
        mltune_params = mql_overrides.env.all_params().get("mltune", {})
        app_params = mql_overrides.env.all_params().get("app", {})

        # Log the logfile location
        logdir = base_params.get('mp_glob_base_log_path', 'logs')
        os.makedirs(logdir, exist_ok=True)
        logger.info(f"Logfile: {logfile}")

        # Log parameter details
        logger.info("Base Parameters:")
        for key, value in base_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Data Parameters:")   
        for key, value in data_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Feature Parameters:")
        for key, value in feat_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("ML Parameters:")
        for key, value in ml_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("ML Tuning Parameters:")
        for key, value in mltune_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("App Parameters:")
        for key, value in app_params.items():
            logger.info(f"  {key}: {value}")
    
        # ----- Load Reference class and time variables -----
        lp_timeframe_name = data_params.get('mp_data_timeframe', 'H4')
        reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
        PRIMARY_SYMBOL = app_params.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD'))
        SECONDARY_SYMBOL = app_params.get('lp_app_secondary_symbol', app_params.get('mp_app_primary_symbol', 'EURCHF'))

        UNIT = reference_config.TIME_CONSTANTS["UNIT"]["SECOND"]
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
            lp_app_primary_symbol=PRIMARY_SYMBOL,
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
        logger.info("Back Window: %s, Forward Window: %s, Prediction Width: %s", back_window, forward_window, pred_width)

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

        # ----- Split Data -----
        seed = mltune_params.get('seed', 42)
        n_samples = len(datafile_X)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        X_train, X_val, X_test, y_train, y_val, y_test = ml_process_config.manual_split_data(datafile_X, datafile_y, train_end, val_end)
        logger.info("Train samples: %s", X_train.shape[0])
        logger.info("Validation samples: %s", X_val.shape[0])
        logger.info("Test samples: %s", X_test.shape[0])

        # ----- Convert to TensorFlow Dataset -----
        tf_batch_size = ml_params.get('tf_batch_size', 32)
        buffer_size = 1000
        train_dataset, val_dataset, test_dataset = ml_process_config.create_simple_tf_dataset(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=tf_batch_size, buffer_size=buffer_size
        )
        logger.info("Train dataset: %s", train_dataset)
        logger.info("Validation dataset: %s", val_dataset)
        logger.info("Test dataset: %s", test_dataset)

        input_shape = train_dataset.element_spec[0].shape
        output_shape = train_dataset.element_spec[1].shape
        logger.info("Input shape: %s", input_shape)
        logger.info("Output shape: %s", output_shape)

        # ----- Model Tuning and Setup -----
        mp_ml_mbase_path = mql_overrides.env.all_params().get("ml_base_path", "./models")
        mp_ml_project_name = mql_overrides.env.all_params().get("ml_project_name", "default_model")
        mp_ml_hard_run = mql_overrides.env.all_params().get("ml_hard_run", False)
        batch_size = ml_params.get("batch_size", 32)
        mp_ml_tf_param_epochs = ml_params.get("epochs", 10)
        ONNX_save = mql_overrides.env.all_params().get("onnx_save", False)
        mp_ml_base_path = mp_ml_mbase_path  # reusing base path
        mp_ml_data_path = mql_overrides.env.all_params().get("ml_data_path", "./data")
        mp_symbol_primary = app_params.get("lp_app_primary_symbol", "EURUSD")
        mp_ml_data_type = mql_overrides.env.all_params().get("ml_data_type", "default")
        
        """
        tuner_config = CMdtuner(
            hypermodel_params=mql_overrides.env.all_params(),
            traindataset=train_dataset,
            valdataset=val_dataset,
            testdataset=test_dataset,
            castmode='float32',
        )
        tuner_config.initialize_tuner()
        best_model = tuner_config.check_and_load_model(mp_ml_mbase_path, ftype='tf')

        mt_obj = tuner_config

        if best_model is None:
            logger.info("No best model loaded. Running tuner search (default run).")
            runtuner = mt_obj.run_search()
            mt_obj.export_best_model(ftype='tf')
        elif mp_ml_hard_run:
            logger.info("Running tuner search (hard run).")
            runtuner = mt_obj.run_search()
            mt_obj.export_best_model(ftype='tf')
        else:
            logger.info("Best model loaded successfully.")
            runtuner = True

        # ----- Train and Evaluate the Model -----
        logger.info("Model: Loading file from directory %s, Model: filename %s", mp_ml_mbase_path, mp_ml_project_name)
        load_model = mt_obj.check_and_load_model(mp_ml_mbase_path, ftype='tf')
        if load_model is not None:
            best_model = load_model
            logger.info("Model: Best model: %s", best_model.name)
            best_model.summary(print_fn=lambda x: logger.info(x))
            tf.keras.backend.clear_session()

            logger.info("Training the best model...")
            best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                batch_size=batch_size,
                epochs=mp_ml_tf_param_epochs
            )
            logger.info("Training completed.")

            logger.info("Evaluating the model...")
            val_metrics = best_model.evaluate(val_dataset, verbose=1)
            test_metrics = best_model.evaluate(test_dataset, verbose=1)
            logger.info("Validation Metrics - Loss: %.4f, Accuracy: %.4f", val_metrics[0], val_metrics[1])
            logger.info("Test Metrics - Loss: %.4f, Accuracy: %.4f", test_metrics[0], test_metrics[1])

            label_scaler.fit(y_train.reshape(-1, 1))
            logger.info("Running predictions and scaling...")
            predicted_fx_price = best_model.predict(test_dataset)
            predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)
            real_fx_price = label_scaler.inverse_transform(y_test.reshape(-1, 1))
            logger.info("Predictions and scaling completed.")

            if np.isnan(real_fx_price).any() or np.isnan(predicted_fx_price).any():
                logger.info("Warning: NaN values found; removing corresponding entries.")
                mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)
                real_fx_price = real_fx_price[mask]
                predicted_fx_price = predicted_fx_price[mask]

            mse = mean_squared_error(real_fx_price, predicted_fx_price)
            mae = mean_absolute_error(real_fx_price, predicted_fx_price)
            r2 = r2_score(real_fx_price, predicted_fx_price)
            logger.info("MSE: %s, MAE: %s, R2: %s", mse, mae, r2)
            logger.info("Mean Squared Error: %s", mse)
            logger.info("Mean Absolute Error: %s", mae)
            logger.info("R2 Score: %s", r2)

            plt.plot(real_fx_price, color='red', label='Real FX Price')
            plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
            plt.title('FX Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('FX Price')
            plt.legend()
            plot_filepath = os.path.join(mp_ml_base_path, 'plot.png')
            plt.savefig(plot_filepath)
            if mql_overrides.env.all_params().get("mp_ml_show_plot", False):
                plt.show()
            logger.info("Plot saved to %s", plot_filepath)

            if ONNX_save:
                mp_output_path = os.path.join(mp_ml_data_path, f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx")
                logger.info("Output Path: %s", mp_output_path)
                opset_version = 17
                spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)
                onnx.save_model(onnx_model, mp_output_path)
                logger.info("Model saved to %s", mp_output_path)
                checker.check_model(onnx_model)
                logger.info("ONNX model is valid.")
                logger.info("ONNX Runtime version: %s", ort.__version__)
            mt5.shutdown()
            logger.info("Finished.")
        else:
            logger.info("No data loaded; exiting.")
            mt5.shutdown()
            logger.info("Finished.")
        """
        
if __name__ == "__main__":
    main(logger)
