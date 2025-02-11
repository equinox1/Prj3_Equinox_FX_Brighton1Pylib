#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

if pchk.mt5 is None or pchk.onnx is None:
    loadmql = False
else:
    mt5 = pchk.mt5
    onnx = pchk.onnx
    loadmql = True


import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import warnings
import posixpath  # For path handling

class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', **kwargs):
        self.kwargs = kwargs
        self.tflog = kwargs.get('tflog', tflog)
        self.warn = kwargs.get('warn', warn)
        self.tfdebug = kwargs.get('tfdebug', False)
        self.precision = kwargs.get('precision', 'mixed_float16')

        
        
        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog
        
        # Set the global policy for mixed precision
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy(self.precision)
        
        print("TensorFlow Version:", tf.__version__)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Running on TPU")
            return tf.distribute.TPUStrategy(tpu)
        except Exception as e:
            print("⚠️ TPU not found, using GPU/CPU:", e)
            return tf.distribute.get_strategy()

    def set_setup_tfdebug(self):
        if self.tfdebug:
            tf.debugging.set_log_device_placement(True)
            tf.config.experimental_run_functions_eagerly(True)
            tf.config.run_functions_eagerly(True)
            tf.config.optimizer.set_jit(True)
            # List available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            print("GPUs Available:", gpus)

            # Enable memory growth to avoid sudden crashes
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.debugging.set_log_device_placement(True)

            if tf.config.list_physical_devices('GPU'):
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                print("Current GPU Memory Usage:", memory_info)
            import psutil
            print("RAM Usage:", psutil.virtual_memory().used / 1e9, "GB")

            import gc
            tf.keras.backend.clear_session()
            gc.collect()



class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        

    def get_params(self):
            return self.__dict__  # Returns all attributes as a dictionary