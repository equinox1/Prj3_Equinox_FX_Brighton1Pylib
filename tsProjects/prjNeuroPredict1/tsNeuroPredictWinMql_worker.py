#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                            tsNeuroPredictWinMql_worker.py        |
# |                        Refactored with CMdtunerSelector          |
# +------------------------------------------------------------------+

from tsMqlSetup import CMqlSetup

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

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
# 🔁 Accept launcher-provided backend override
env_backend = os.environ.get("MLTUNE_BACKEND", "tensorflow")
env_gtuner = os.environ.get("GTUNER_MODEL", env_backend)

mql_overrides.env.override_params({
    "mltune": {"backend": env_backend},
    "app": {"gtuner_model": env_gtuner}
})


app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup ----


# --- Strategy & Platform ---
strategy = setup_config.get_computation_strategy()
platform = run_platform.RunPlatform()
logger.info("Detected platform: %s | MetaTrader5 active: %s", platform_checker.get_platform(), platform.check_mql_state())

import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

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
    oracle = OracleClient(host=xerces_server, port=xerces_port)

    # Update input shape and basic tuning params
    mltune = hyperparams.setdefault('mltune', {})
    mltune.update({
        'data_input_shape': input_shape,
        'input_shape': input_shape,
        'input_width': mltune.get('input_width', 24),
        'shift': mltune.get('shift', 24)
    })

    backend = os.environ.get("MLTUNE_BACKEND", "tensorflow").lower()
    gtuner_model = os.environ.get("GTUNER_MODEL", backend).lower()

    logger.info(f"Worker Using GTuner model: {gtuner_model}")
    logger.info(f"Worker Using backend: {backend}")

    if backend == "pytorch":
        from tsMqlMLTuner.tsMqlMLTunerModTorch import PyTorchTuner
        tuner = PyTorchTuner(
            oracle=oracle,
            hypermodel_params=hyperparams,
            traindataset=(X, y),
            valdataset=(X, y)  # Optional: add real split later
        )
        logger.info("Worker running PyTorch tuner loop...")
        tuner.run_search()
        return

    elif backend == "tensorflow":
        import tensorflow as tf
        buffer_size = 10000
        batch_size = 32
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        traindataset = valdataset = testdataset = dataset

        from tsMqlMLTuner.cm_dtuner_selector import CMdtunerSelector
        tuner = CMdtunerSelector(
            oracle=oracle,
            hypermodel_params=hyperparams,
            traindataset=traindataset,
            valdataset=valdataset,
            testdataset=testdataset,
            castmode='float32'
        )
        logger.info("Worker running TensorFlow tuner loop...")
        tuner.run_search()

    else:
        raise ValueError(f"Unsupported backend: {backend}")



if __name__ == "__main__":
    X, y, shape, params = main(logger)
    run_worker_loop(X, y, shape, params)
    logger.info("Worker process completed.")