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
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
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
from tsMqlMLTuner import OracleClient, CMdtunerSelector

# --- Environment Setup ---
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TUNER_ID"] = "worker"

setup = CMqlSetup(loglevel='INFO', warn='ignore', precision='mixed_bfloat16', tfdebug=False, num_cores=48, num_threads=4)
server = '192.168.1.103'
port = 9000
logdir, logfile = setup.set_log_dir(logfile='tsneuropredict_app.log', servername="WINSVRXERCES01")

# --- Logger Setup ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
try:
    fh = logging.FileHandler(logfile, mode='w', encoding='utf-8')
except OSError:
    fh = logging.FileHandler('fallback.log', mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.info("Worker logging configured. Logfile: %s", logfile)

# --- Strategy & Platform ---
strategy = setup.get_computation_strategy()
platform = run_platform.RunPlatform()
logger.info("Detected platform: %s | MetaTrader5 active: %s", platform_checker.get_platform(), platform.check_mql_state())

# --- Main Worker Routine ---
def main(logger):
    utils = CUtilities()
    overrides = CMqlOverrides()

    # Default parameter setup
    overrides.env.override_params({"app": {'mp_app_ml_hard_run': False}})
    overrides.env.override_params({"mltune": {'batch_size': 8}})
    overrides.env.override_params({"data": {'mp_data_timeframe': mt5.TIMEFRAME_H4}})

    base = overrides.env.all_params().get("base", {})
    data = overrides.env.all_params().get("data", {})
    ml = overrides.env.all_params().get("ml", {})
    mltune = overrides.env.all_params().get("mltune", {})
    app = overrides.env.all_params().get("app", {})

    timeframe = data.get("mp_data_timeframe", mt5.TIMEFRAME_H4)
    ref = CMqlRefConfig(loaded_data_type="MINUTE", required_data_type=timeframe)
    timeconst = ref.TIME_CONSTANTS[0] if isinstance(ref.TIME_CONSTANTS, list) else ref.TIME_CONSTANTS

    now = ref.get_current_time()
    utc_from = CDataLoader().set_mql_timezone(now["CURRENTYEAR"] - 1, now["CURRENTMONTH"], now["CURRENTDAY"], now["TIMEZONE"])
    utc_to   = CDataLoader().set_mql_timezone(now["CURRENTYEAR"], now["CURRENTMONTH"], now["CURRENTDAY"], now["TIMEZONE"])

    CMqlBrokerConfig(app.get("mp_app_broker")).run_mql_login()

    dataloader = CDataLoader(
        lp_utc_from=utc_from,
        lp_utc_to=utc_to,
        lp_timeframe=timeframe,
        lp_app_primary_symbol=app.get("mp_app_primary_symbol", "EURUSD"),
        lp_app_rows=data.get("mp_data_rows", 1000),
        lp_app_rowcount=data.get("mp_data_rowcount", 10000)
    )

    df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = dataloader.run_dataloader_services()
    df_rates = CDataProcess(mp_unit=timeconst["UNIT"]["SECOND"]).run_dataprocess_services(df=df_file_rates, df_name='df_file_rates')

    X_raw, y = CDMLProcess().Create_Xy_input_and_target(df_rates, back_window=24, forward_window=24,
                                                        features=[ml.get("mp_ml_input_keyfeat", "Close")])

    # Scale
    nsamples, nsteps, nfeatures = X_raw.shape
    X_flat = X_raw.reshape((nsamples * nsteps, nfeatures))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(X_raw.shape)

    if X_scaled.ndim == 4:
        X_scaled = np.squeeze(X_scaled, axis=-1)
    elif X_scaled.ndim == 2:
        X_scaled = np.expand_dims(X_scaled, axis=-1)

    input_shape = X_scaled.shape[1:]
    logger.info("Data loaded and shaped. Input shape: %s | Output: %s", input_shape, y.shape)

    return X_scaled, y, input_shape, overrides.env.all_params()

# --- Trial Execution Loop (via Selector) ---
def run_worker_loop(X, y, input_shape, hyperparams):
    oracle = OracleClient(host=server, port=port)

    # Patch override shapes if needed
    hyperparams['mltune']['data_input_shape'] = input_shape
    hyperparams['mltune']['input_shape'] = input_shape

    tuner = CMdtunerSelector(
        oracle=oracle,
        hypermodel_params=hyperparams,
        traindataset=(X, y),
        valdataset=(X, y),
        testdataset=(X, y),
        castmode='float32'
    )
    logger.info("Worker tuner initialized. Waiting for Oracle trials.")
    tuner.run()

if __name__ == "__main__":
    X, y, shape, params = main(logger)
    run_worker_loop(X, y, shape, params)
    logger.info("Worker process completed.")
