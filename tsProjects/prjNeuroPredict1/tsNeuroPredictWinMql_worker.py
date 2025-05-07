#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                tsNeuroPredictWinMql_worker.py    |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+

# --- [ imports ] ---
import os
import logging
from cv2 import log
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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
from tsMqlMLTuner import CMdtuner

#Oracle imports
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense



# --- [ environment setup ] ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


os.environ["TUNER_ID"] = "worker"
tuner_id = os.environ.get("TUNER_ID", "worker")
setup_config = CMqlSetup(loglevel='INFO', warn='ignore', precision='mixed_bfloat16', tfdebug=False, num_cores=48, num_threads=4)
xerces_servername = "WINSVRXERCES01"
xerces_server = '192.168.1.103'
xerces_port = 9000
xerces_logfile = 'tsneuropredict_app.log'
global_logdir,global_logfile=setup_config.set_log_dir(logdir=None,logfile=xerces_logfile, servername=xerces_servername)
print(f"Logdir: {global_logdir}")
# --- [ logger setup ] ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

try:
    fh = logging.FileHandler(global_logfile, mode='w', encoding='utf-8')
except OSError as e:
    print(f"Error creating log file: {e}")
    fh = logging.FileHandler('fallback.log', mode='w', encoding='utf-8')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.info("Logging configured successfully.")

# --- [ strategy and overrides ] ---
strategy = setup_config.get_computation_strategy()
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, MetaTrader5 load state: {loadmql}")

# --- [ main workflow ] ---
def main(logger):
    utils_config = CUtilities()
    mql_overrides = CMqlOverrides()

    base_params = mql_overrides.env.all_params().get("base", {})
    data_params = mql_overrides.env.all_params().get("data", {})
    ml_params = mql_overrides.env.all_params().get("ml", {})
    mltune_params = mql_overrides.env.all_params().get("mltune", {})
    app_params = mql_overrides.env.all_params().get("app", {})



     # ----- Model Tuning and Setup -----
    mql_overrides.env.override_params({"app": {'mp_app_ml_hard_run': False}})
    mql_overrides.env.override_params({"mltune": {'batch_size': 8}})
    mql_overrides.env.override_params({"data": {'mp_data_timeframe': 'mt5.TIMEFRAME_H4'}})
    logger.info("Main: mp_app_ml_hard_run: %s", app_params.get('mp_app_ml_hard_run', True))
    logger.info("Main: mp_ml_mbase_path: %s", base_params.get('mp_glob_base_ml_project_dir', None))
    logger.info("Main: batch_size: %s", base_params.get('batch_size', None))
    logger.info("Main: mp_data_timeframe: %s", data_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_H4'))

    lp_timeframe_name = data_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_H4')
    logger.info("Worker Timeframe: %s", lp_timeframe_name)
    reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
    time_constants = reference_config.TIME_CONSTANTS[0] if isinstance(reference_config.TIME_CONSTANTS, list) else reference_config.TIME_CONSTANTS

    HOUR = reference_config.get_timevalue('HOUR')
    CURRENT_TIME = reference_config.get_current_time()
    TIMEZONE = CURRENT_TIME["TIMEZONE"]
    CURRENTYEAR, CURRENTMONTH, CURRENTDAY = CURRENT_TIME["CURRENTYEAR"], CURRENT_TIME["CURRENTMONTH"], CURRENT_TIME["CURRENTDAY"]

    broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
    broker_status = broker_config.run_mql_login()
    logger.info("MetaTrader5 login status: %s", broker_status)

    mp_data_history_size = data_params.get('mp_data_history_size', 1)
    data_loader = CDataLoader(
        lp_utc_from=CDataLoader().set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAY, TIMEZONE),
        lp_utc_to=CDataLoader().set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAY, TIMEZONE),
        lp_timeframe=lp_timeframe_name,
        lp_app_primary_symbol=app_params.get('mp_app_primary_symbol', 'EURUSD'),
        lp_app_rows=data_params.get('mp_data_rows', 1000),
        lp_app_rowcount=data_params.get('mp_data_rowcount', 10000)
    )

    df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = data_loader.run_dataloader_services()
    datafile = df_file_rates

    data_process = CDataProcess(mp_unit=time_constants["UNIT"]["SECOND"])
    datafile = data_process.run_dataprocess_services(df=datafile, df_name='df_file_rates')

    ml_process = CDMLProcess()
    back_window, forward_window, pred_width = ml_process.create_ml_window(timeval=HOUR)
    data_X, data_y = ml_process.Create_Xy_input_and_target(
        datafile, back_window=back_window, forward_window=forward_window,
        features=[ml_params.get("mp_ml_input_keyfeat", "Close")]
    )

   
    # Scaling
    nsamples, nsteps, nfeatures = data_X.shape
    X_reshaped = data_X.reshape((nsamples * nsteps, nfeatures))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    datafile_X_scaled = X_scaled.reshape(data_X.shape)
    if datafile_X_scaled.ndim == 4:
        datafile_X_scaled = np.squeeze(datafile_X_scaled, axis=-1)
    elif datafile_X_scaled.ndim == 2:
        datafile_X_scaled = np.expand_dims(datafile_X_scaled, axis=-1)

    logger.info("Prepared scaled input data for tuning. Shape: %s", datafile_X_scaled.shape)
    return datafile_X_scaled

def build_and_train_model(hparams, input_shape, logger):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(hparams.get('dense_1_units', 64), activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=hparams.get('optimizer', 'adam'),
        loss=hparams.get('loss', 'mse'),
        metrics=[hparams.get('metric', 'mse')]
    )

    x_train = np.random.rand(100, *input_shape)
    y_train = np.random.rand(100)
    history = model.fit(x_train, y_train, epochs=hparams.get('epochs', 5), batch_size=8, verbose=0, validation_split=0.2)
    return history.history['val_loss'][-1]

def run_oracle_client_loop(datafile_X_scaled, logger):
    oracle_client = OracleClient(host=xerces_server, port=xerces_port)
    input_shape = datafile_X_scaled.shape[1:]
    logger.info(f"Worker input shape: {input_shape}")

    while True:
        try:
            logger.info("Requesting trial from OracleServer...")
            trial_data = oracle_client.get_trial()
            trial_id = trial_data['trial_id']
            hparams = trial_data['hyperparameters']
            logger.info(f"Received Trial {trial_id}: {hparams}")
            val_loss = build_and_train_model(hparams, input_shape, logger)
            logger.info(f"Reporting result for Trial {trial_id}: val_loss={val_loss}")
            oracle_client.report_trial_result(trial_id, val_loss)
            oracle_client.update_trial_status(trial_id, "COMPLETED")
        except Exception as e:
            logger.error(f"OracleClient error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    scaled_X = main(logger)
    run_oracle_client_loop(scaled_X, logger)
    logger.info("Worker process completed.")