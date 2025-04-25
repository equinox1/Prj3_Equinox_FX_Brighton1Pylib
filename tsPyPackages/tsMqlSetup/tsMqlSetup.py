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
        #os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
       
        logger.info(f"TF_CPP_MIN_LOG_LEVEL: {os.environ['TF_CPP_MIN_LOG_LEVEL']}")
        #logger.info(f"TF_ENABLE_ONEDNN_OPTS: {os.environ['TF_ENABLE_ONEDNN_OPTS']}")

        
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
        logger.info(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
        logger.info(f"TF_NUM_INTRAOP_THREADS: {os.environ['TF_NUM_INTRAOP_THREADS']}")
        logger.info(f"TF_NUM_INTEROP_THREADS: {os.environ['TF_NUM_INTEROP_THREADS']}")
        logger.info(f"MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
        logger.info(f"KMP_BLOCKTIME: {os.environ['KMP_BLOCKTIME']}")
        logger.info(f"KMP_SETTINGS: {os.environ['KMP_SETTINGS']}")
        logger.info(f"KMP_AFFINITY: {os.environ['KMP_AFFINITY']}")
        logger.info(f"KMP_DUPLICATE_LIB_OK: {os.environ['KMP_DUPLICATE_LIB_OK']}")
        logger.info(f"KMP_INIT_WAIT_TIMEOUT: {os.environ['KMP_INIT_WAIT_TIMEOUT']}")
        logger.info(f"KMP_WARNINGS: {os.environ['KMP_WARNINGS']}")
        logger.info(f"KMP_FORCE_USE_OPENMP: {os.environ['KMP_FORCE_USE_OPENMP']}")
        logger.info(f"KMP_USE_ITT_NOTIFY: {os.environ['KMP_USE_ITT_NOTIFY']}")
        

        # Set the number of threads for TensorFlow operations
        tf.config.threading.set_intra_op_parallelism_threads(self.num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(self.num_cores)
        tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra) for performance optimization
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})  # Enable mixed precision
        tf.config.optimizer.set_experimental_options({"layout_optimizer": True})  # Enable layout optimizer for performance
        logger.info(f"TensorFlow Intra Op Parallelism Threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
        logger.info(f"TensorFlow Inter Op Parallelism Threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TensorFlow XLA Enabled: {tf.config.optimizer.get_jit()}")
        logger.info(f"TensorFlow Mixed Precision Enabled: {tf.config.optimizer.get_experimental_options().get('auto_mixed_precision')}")
        logger.info(f"TensorFlow Layout Optimizer Enabled: {tf.config.optimizer.get_experimental_options().get('layout_optimizer')}")
        logger.info(f"TensorFlow MKL Enabled: {tf.config.optimizer.get_experimental_options().get('mkl')}")
        logger.info(f"TensorFlow OneDNN Enabled: {tf.config.optimizer.get_experimental_options().get('onednn')}")


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
            logger.info(f"GPUs Available: {gpus}")

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

    def get_computation_strategy_base(self):
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                print("✅ Running on TPU")
                return tf.distribute.TPUStrategy(tpu)
            except ValueError: # Catch the specific error when TPU is not found
                print("⚠️ TPU not found, using GPU/CPU")
                return tf.distribute.get_strategy()

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Running on TPU")
            return tf.distribute.TPUStrategy(tpu)
        except (ValueError, tf.errors.NotFoundError) as e:
            print(f"⚠️ TPU not found or initialization failed: {e}")
            print("⚠️ Trying MultiWorkerMirroredStrategy")

        try:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print("✅ Running on MultiWorker GPU/CPU")
            return strategy
        except (tf.errors.InternalError, tf.errors.UnavailableError) as e:
            print(f"⚠️ MultiWorker strategy failed: {e}")
            print("⚠️ Falling back to default strategy")

        # Fallback
        try:
            strategy = tf.distribute.get_strategy()
            print("⚠️ Using default strategy (likely CPU)")
            return strategy
        except Exception as e:
            print(f"❌ Failed to initialize any strategy: {e}")
            raise RuntimeError("No valid computation strategy could be initialized.")


    def set_log_dir(self, logdir=None,logfile= 'tslog', servername=None):
        import socket
        self.hostname = os.getenv('HOSTNAME', socket.gethostname())
        print(f"Set log:Hostname: {self.hostname}")
        print(f"Set log:Servername: {servername}")
        print(f"Set log: os_platform: {os_platform}")

        if logdir is None:
            if self.hostname == servername and os_platform == 'Windows':
                self.global_logdir = os.path.join(r'C:', '\\', 'WinRunMnt1', '8.0 Projects', '8.3 ProjectModelsEquinox', 'EQUINRUN', 'Logdir')
                self.global_logfile = os.path.join(self.global_logdir, logfile)
                if not os.path.exists(self.global_logdir):
                    os.makedirs(self.global_logdir)
                if not os.path.exists(self.global_logfile):
                    with open(self.global_logfile, 'w') as f:
                        f.write("Log file created successfully.")
            elif os_platform == 'Linux':
                self.global_logdir = os.path.join('/mnt', '8.0 Projects', '8.3 ProjectModelsEquinox', 'EQUINRUN', 'Logdir')
                self.global_logfile = os.path.join(self.global_logdir, logfile)
                if not os.path.exists(self.global_logdir):
                    os.makedirs(self.global_logdir)
                if not os.path.exists(self.global_logfile):
                    with open(self.global_logfile, 'w') as f:
                        f.write("Log file created successfully.")
            elif os_platform == 'Darwin':
                self.global_logdir = os.path.join('/Users', 'shepa', 'OneDrive', '8.0 Projects', '8.3 ProjectModelsEquinox', 'EQUINRUN', 'Logdir')
                self.global_logfile = os.path.join(self.global_logdir, logfile)
                if not os.path.exists(self.global_logdir):
                    os.makedirs(self.global_logdir)
                if not os.path.exists(self.global_logfile):
                    with open(self.global_logfile, 'w') as f:
                        f.write("Log file created successfully.")
            else:
                self.global_logdir = os.path.join(r'C:', '\\','/Users', 'shepa', 'OneDrive', '8.0 Projects', '8.3 ProjectModelsEquinox', 'EQUINRUN', 'Logdir')
                self.global_logfile = os.path.join(self.global_logdir, logfile)
                if not os.path.exists(self.global_logdir):
                    os.makedirs(self.global_logdir) 
                if not os.path.exists(self.global_logfile):
                    with open(self.global_logfile, 'w') as f:
                        f.write("Log file created successfully.")
        
        return self.global_logdir,self.global_logfile

    def set_logger(self, global_logfile):
            # Set up the logger
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            if logger.hasHandlers():
                logger.handlers.clear()
    
            try:
                # Specify encoding='utf-8' in FileHandler
                fh = logging.FileHandler(global_logfile, mode='w', encoding='utf-8')
            except OSError as e:
                print(f"Error creating log file: {e}")
                fh = logging.FileHandler('fallback.log', mode='w', encoding='utf-8')
                # Fallback to a local log file
                print("Fallback log file created: fallback.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info("Logging configured successfully with FileHandler.")
            logger.info("Logfile: %s", global_logfile)
            return logger


