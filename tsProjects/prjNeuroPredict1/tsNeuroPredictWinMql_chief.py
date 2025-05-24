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
from tsMqlMLProcess import CDMLProcess
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

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
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

        # Check if tuning was successful
        oracle = OracleClient(host=xerces_server, port=xerces_port)
        best_trial = oracle.get_best_trial()
        if not best_trial or "hyperparameters" not in best_trial:
            logger.error("❌ No best trial found. Skipping training/export.")
            sys.exit(1)

        logger.info(f"🏆 Best trial ID: {best_trial['trial_id']}")
        hp = best_trial["hyperparameters"]
        # Rebuild tuner for post-tuning processing
        tuner_config = CMdtunerSelector(
            oracle=oracle,  # reuse or reconstruct from oracle_dir
            hypermodel_params=hyperparams,
            traindataset=train_dataset,
            valdataset=val_dataset,
            testdataset=test_dataset,
            castmode='float32'
        )

        # Get best trial
        best_trial = tuner_config.oracle.get_best_trial()
        if not best_trial:
            logger.error("No best trial found. Cannot continue with post-tuning training.")
            sys.exit(1)

        logger.info(f"🏆 Best trial selected: {best_trial['trial_id']}")
        hp = best_trial['hyperparameters']

        # Build and fit best model
        best_model = tuner_config.build_model(hp)
        logger.info("Fitting model on full training set...")

        # --- FIX: Properly call fit() and set up callbacks ---
        batch_size = hp.get("batch_size", 32)
        epochs = hp.get("epochs", 10)
        os.makedirs(logdir, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=logdir),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]
        history = best_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        # === FINAL PREDICTIONS AND MSE ===
        logger.info("📈 Predicting on test set using best model...")
        try:
            y_pred = best_model.predict(test_dataset)
            y_true = np.concatenate([y for _, y in test_dataset], axis=0)

            test_mse = np.mean(np.square(y_pred.flatten() - y_true.flatten()))
            logger.info(f"📉 Final Test MSE: {test_mse:.6f}")

            # Optional: Save prediction plot
            plt.figure(figsize=(12, 5))
            plt.plot(y_true, label="True")
            plt.plot(y_pred, label="Predicted")
            plt.legend()
            plt.title("Prediction vs Ground Truth")
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
        symbol_name = app_params.get('mp_app_primary_symbol', 'EURUSD')
        model_path = os.path.join(modeldatapath, f"{modelname}.h5")
        best_model.save(model_path)
        logger.info(f"✅ Model saved: {model_path}")

        # Optional: Convert to ONNX
        if app_params.get("mp_app_ONNX_save", False):
            import tf2onnx
            import onnx
            spec = [tf.TensorSpec(best_model.input_shape, tf.float32, name="input")]
            onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=17)
            onnx_path = os.path.join(modeldatapath, f"model_{symbol_name}_data.onnx")
            onnx.save_model(onnx_model, onnx_path)
            logger.info(f"🧠 ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    main(logger)