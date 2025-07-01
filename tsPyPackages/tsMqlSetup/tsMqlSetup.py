# tsMqlSetup.py
import os
import warnings
import gc
import logging
import socket
import sys
import codecs
import io
from pathlib import Path
from loguru import logger as loguru_logger
import colorlog

# Set fixed TF environment variables
os.environ.update({
    "TF_FORCE_UNIFIED_MEMORY": "1",
    "TF_DISABLE_POOL_ALLOCATOR": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "KMP_DUPLICATE_LIB_OK": "True"
})

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf  # Import here before Keras or mixed_precision

# ⛔ Must set threading before using Keras or anything that initializes TF runtime
def preconfigure_tf_threads(cores=48, threads=8):
    total_threads = cores * threads
    try:
        tf.config.threading.set_intra_op_parallelism_threads(total_threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    except RuntimeError as e:
        print(f"[WARNING] Threading setup skipped: {e}", file=sys.stderr)

# Call early
preconfigure_tf_threads()

# Now it's safe to import mixed_precision, Keras, etc.
from tensorflow.keras.mixed_precision import Policy
from tsMqlPlatform import run_platform, platform_checker
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

class CMqlSetup:
    _log_setup_done = False

    def __init__(self, loglevel='DEBUG', tflog=2, warn='ignore', precision='mixed_float16',
                 tfdebug=False, num_cores=48, num_threads=8, **kwargs):
        self.tflog = tflog
        self.loglevel = loglevel.upper()
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores = num_cores
        self.num_threads = num_threads
        self.sumthreads = self.num_cores * self.num_threads
        self.kwargs = kwargs

        self._setup_warnings()
        self._setup_tf_logging()
        self._set_precision_policy()
        self._configure_debug()

    def _setup_warnings(self):
        warnings.filterwarnings(self.warn)

    def _setup_tf_logging(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.tflog)

    def _set_precision_policy(self):
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))

    def _configure_debug(self):
        if not self.tfdebug:
            return
        tf.debugging.set_log_device_placement(True)
        tf.config.run_functions_eagerly(True)
        tf.config.optimizer.set_jit(False)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            logging.info(f"GPUs available: {gpus}")
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                logging.info(f"GPU Memory Info: {mem_info}")
        except Exception as e:
            logging.warning(f"Could not retrieve GPU memory info: {e}")
        import psutil
        logging.info(f"RAM Used: {psutil.virtual_memory().used / 1e9:.2f} GB")
        tf.keras.backend.clear_session()
        gc.collect()

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            logging.info("Using TPU")
            return tf.distribute.TPUStrategy(tpu)
        except Exception:
            pass
        for strategy_cls, label in [
            (tf.distribute.MultiWorkerMirroredStrategy, "MultiWorker GPU/CPU"),
            (tf.distribute.MirroredStrategy, "Mirrored GPU/CPU"),
            (lambda: tf.distribute.OneDeviceStrategy("/cpu:0"), "CPU (OneDevice)"),
            (tf.distribute.experimental.ParameterServerStrategy, "Parameter Server"),
            (tf.distribute.experimental.CentralStorageStrategy, "Central Storage"),
        ]:
            try:
                strategy = strategy_cls()
                logging.info(f"Using {label}")
                return strategy
            except Exception as e:
                logging.warning(f"{label} failed: {e}")
        raise RuntimeError("No valid strategy available.")
