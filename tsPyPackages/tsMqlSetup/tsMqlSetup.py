import os
import warnings
import gc
import logging

import socket
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

from tsMqlPlatform import run_platform, platform_checker
from rich.logging import RichHandler

# -- Base Env Setup --
os.environ.update({
    "TF_FORCE_UNIFIED_MEMORY": "1",
    "TF_DISABLE_POOL_ALLOCATOR": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "TF_DISABLE_MKL": "1",
    "TF_CPU_ALLOCATOR_MAX_BYTES": str(128 * 1024**3),
    "TF_GPU_ALLOCATOR_MAX_BYTES": str(128 * 1024**3),
})

# -- Platform Info --
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()




class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', precision='mixed_float16', tfdebug=False, num_cores=28, num_threads=2, **kwargs):

        self.tflog = tflog
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores = num_cores  # Physical cores
        self.num_threads = num_threads
        self.sumthreads = self.num_cores * self.num_threads
        self.gtuner_model = kwargs.get('gtuner_model', 'tensorflow')
        self.kwargs = kwargs

        self._setup_warnings()
        self._setup_tf_logging()
        self._set_precision_policy()
        self._configure_tf()
        self._configure_debug()
        

    def _setup_warnings(self):
        warnings.filterwarnings(self.warn)

    def _setup_tf_logging(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog

    def _set_precision_policy(self):
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))

    def _configure_tf(self):
        os.environ.update({
            "OMP_NUM_THREADS": str(self.sumthreads),
            "TF_NUM_INTRAOP_THREADS": str(self.sumthreads),
            "TF_NUM_INTEROP_THREADS": str(self.num_threads),
            "MKL_NUM_THREADS": str(self.sumthreads),
            "KMP_BLOCKTIME": "1",
            "KMP_SETTINGS": "1",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_DUPLICATE_LIB_OK": "True",
            "KMP_INIT_WAIT_TIMEOUT": "2000",
            "KMP_WARNINGS": "0",
            "KMP_FORCE_USE_OPENMP": "1",
            "KMP_USE_ITT_NOTIFY": "0"
        })

        tf.config.threading.set_intra_op_parallelism_threads(self.sumthreads)
        tf.config.threading.set_inter_op_parallelism_threads(self.num_threads)

        tf.config.optimizer.set_experimental_options({
            "auto_mixed_precision": True,
            "layout_optimizer": True,
            "mkl": False,
            "onednn": False
        })

        self._enable_gpu_memory_growth()

    def _enable_gpu_memory_growth(self):
        try:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Failed to set memory growth: {e}")

    def _configure_debug(self):
        if not self.tfdebug:
            return

        tf.debugging.set_log_device_placement(True)
        tf.config.run_functions_eagerly(True)
        tf.config.optimizer.set_jit(False)

        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPUs available: {gpus}")

        if gpus:
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                print("GPU Memory Info:", mem_info)
            except Exception as e:
                logger.warning(f"GPU memory info not available: {e}")

        import psutil
        print("RAM Used:", psutil.virtual_memory().used / 1e9, "GB")

        tf.keras.backend.clear_session()
        gc.collect()

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Using TPU")
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
                print(f"✅ Using {label}")
                return strategy
            except Exception as e:
                logger.warning(f"{label} failed: {e}")

        raise RuntimeError("❌ No valid strategy available.")

    def set_log_dir1(self, logdir=None, logfile='tslog', servername=None, ltuner=None):
        import socket
        hostname = os.getenv('HOSTNAME', socket.gethostname())
        print(f"Hostname: {hostname}")

        if logdir is None:
            if hostname == servername and os_platform == 'Windows':
                base_path = r'C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir'
            elif os_platform == 'Linux':
                base_path = '/mnt/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            elif os_platform == 'Darwin':
                base_path = '/Users/shepa/OneDrive/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            else:
                base_path = os.path.expanduser('~/EQUINRUN/Logdir')  # ✅ fallback

            subdir = servername if servername else '_unknown'
            base_path = os.path.join(base_path, self.gtuner_model, subdir)
        else:
            base_path = logdir

        os.makedirs(base_path, exist_ok=True)

        # Prevent nested subfolders from logfile name
        logfile_name = os.path.basename(logfile)
        self.global_logdir = base_path
        self.global_logfile = os.path.join(base_path, logfile_name)
       

        # Make sure the directory exists
        os.makedirs(os.path.dirname(self.global_logfile), exist_ok=True)

        try:
            with open(self.global_logfile, 'a') as f:
                f.write('')  # Create an empty file to ensure it is writable
        except Exception as e:
            print(f"Could not create logfile at {self.global_logfile}: {e}")
            raise

        return self.global_logdir, self.global_logfile



    def setup_global_logger1(self, logdir=None, logfile='tslog', servername=None, ltuner=None):
        logger = logging.getLogger()
        if logger.hasHandlers():
            return logger  # Prevent duplicate setup

        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfilein, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.info(f"Logger initialized with output file: {logfilein}")
        return logger



    def set_log_dir(self, logdir=None, logfile='tslog', servername=None, ltuner=None):
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")

        # Default logdir construction based on platform
        if logdir is None:
            if hostname == servername and os_platform == 'Windows':
                base_path = r'C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir'
            elif os_platform == 'Linux':
                base_path = '/mnt/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            elif os_platform == 'Darwin':
                base_path = '/Users/shepa/OneDrive/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            else:
                base_path = os.path.expanduser('~/EQUINRUN/Logdir')

        else:
            base_path = logdir

        # Final log directory: Logdir / Hostname / ltuner
        final_logdir = os.path.join(base_path, hostname, ltuner)
        os.makedirs(final_logdir, exist_ok=True)

        self.global_logdir = final_logdir
        self.global_logfile = os.path.join(final_logdir, 'tsneuropredict_app.log')

        try:
            with open(self.global_logfile, 'a') as f:
                f.write('')
        except Exception as e:
            print(f"Could not create logfile at {self.global_logfile}: {e}")
            raise

        return self.global_logdir, self.global_logfile



    def setup_global_logger(self, logfilein=None):
        if logfilein is None:
            logfilein = getattr(self, 'global_logfile', 'tsneuropredict_app.log')

        logger = logging.getLogger()
        if logger.hasHandlers():
            return logger  # Avoid duplicate handlers

        logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(logfilein, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Rich console handler
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_formatter = logging.Formatter('%(message)s')
        rich_handler.setFormatter(console_formatter)
        logger.addHandler(rich_handler)

        logger.info(f"Logger initialized with file: {logfilein}")
        return logger
