
import os
import posixpath  # For path handling
import warnings
import gc

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', precision='mixed_float16', tfdebug=False, **kwargs):
        self.tflog = tflog
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.kwargs = kwargs

        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog

        # Set the global policy for mixed precision
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))

        print(f"TensorFlow Version: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

        self.set_setup_tfdebug()  # Call debugging setup if enabled

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Running on TPU")
            return tf.distribute.TPUStrategy(tpu)
        except ValueError: # Catch the specific error when TPU is not found
            print("⚠️ TPU not found, using GPU/CPU")
            return tf.distribute.get_strategy()

    def set_setup_tfdebug(self):
        if self.tfdebug:
            tf.debugging.set_log_device_placement(True)
            tf.config.experimental_run_functions_eagerly(True)
            tf.config.run_functions_eagerly(True)  # This might not be needed in TF 2.x
            tf.config.optimizer.set_jit(True)

            gpus = tf.config.list_physical_devices('GPU')
            print("GPUs Available:", gpus)

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.debugging.set_log_device_placement(True)  # Already set above

            if gpus:
                try: # Add try-except to handle potential errors
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    print("Current GPU Memory Usage:", memory_info)
                except Exception as e:
                    print(f"Error getting GPU memory info: {e}")

            import psutil
            print("RAM Usage:", psutil.virtual_memory().used / 1e9, "GB")

            tf.keras.backend.clear_session()
            gc.collect()


