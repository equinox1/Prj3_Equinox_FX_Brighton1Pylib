"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlSetup.py
File: tsPyPackages/tsMqlSetup/tsMqlSetup.py
Description: Setup initial values
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

import os
import posixpath  # For path handling
import warnings
import gc
import logging

import tensorflow as tf
import intel_tensorflow as itex
from tensorflow.keras.mixed_precision import Policy

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES,  get_config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()

logger = logging.getLogger(__name__)

logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', precision='mixed_float16', tfdebug=False,num_cores=24,num_threads = 2, **kwargs):
        self.tflog = tflog
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores=num_cores # Number of CPU cores to use def 24
        self.kwargs = kwargs

        # Set the TensorFlow logging level
        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog
        
        print(f"TensorFlow Version: {tf.__version__}")
        # Set the global policy for mixed precision
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))
        # Set the GPU switch
        self.__set_gpu_memory_growth()  # Set GPU memory growth
     
         # Set the TF Debug
        self.__set_setup_tfdebug()  # Call debugging setup if enabled
        # Set Multi-threading
        self.__set_multi_threading()
    
       

    def __set_multi_threading(self):
        # Set the number of threads for OpenMP and TensorFlow operations
        os.environ["OMP_NUM_THREADS"] = str(self.num_cores)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(self.num_cores)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(self.num_cores)
        os.environ["MKL_NUM_THREADS"] = str(self.num_cores)
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["KMP_INIT_WAIT_TIMEOUT"] = "2000"
        os.environ["KMP_WARNINGS"] = "0"
        os.environ["KMP_FORCE_USE_OPENMP"] = "1"
        os.environ["KMP_USE_ITT_NOTIFY"] = "0"
        
        # Set the number of threads for TensorFlow operations
        tf.config.threading.set_intra_op_parallelism_threads(self.num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(self.num_cores)
        tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra) for performance optimization
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})  # Enable mixed precision
        tf.config.optimizer.set_experimental_options({"layout_optimizer": True})  # Enable layout optimizer for performance

    def __set_gpu_memory_growth(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def __set_setup_tfdebug(self):
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

