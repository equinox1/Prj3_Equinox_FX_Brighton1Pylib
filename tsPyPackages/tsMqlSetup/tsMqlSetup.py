import os
import warnings
import gc
import logging
import socket
import codecs
import io
import sys

os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tsMqlPlatform import run_platform, platform_checker
from rich.logging import RichHandler
from rich.traceback import install

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

class CMqlSetup:
    def __init__(self, loglevel='DEBUG', tflog =2,warn='ignore', precision='mixed_float16', tfdebug=False,num_cores=48, num_threads=8 ,**kwargs):
        self.tflog = tflog
        self.loglevel = loglevel.upper()
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores = num_cores
        self.num_threads = num_threads
        self.sumthreads = self.num_cores * self.num_threads
        self.gtuner_model = kwargs.get('gtuner_model', 'tensorflow')
        self.kwargs = kwargs

        self._setup_warnings()
        self._setup_tf_logging()
        self._set_precision_policy()
        self._configure_tf()
        self._configure_debug()
        # The _encoding method is called from setup_logging for console handler configuration.
        # self._encoding() # Removed from here, as setup_logging will handle it more directly.

    def _encoding(self):
        # This method's logic is now more directly handled within setup_logging's RichHandler config
        # For general system-wide UTF-8 setting, it's still useful.
        if sys.platform.startswith('win'):
            if sys.getfilesystemencoding() != 'utf-8':
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
                # Ensure the console itself uses UTF-8 if it's a new one.
                # This line sets the console codepage for the current process.
                os.system('chcp 65001')
                # Reconfigure stdout/stderr for Python's own stream handling
                # This might not affect RichHandler's direct stream, but good practice.
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
                # Register a lenient error handler for codecs
                codecs.register_error('strict', codecs.ignore_errors)


    def _setup_warnings(self):
        warnings.filterwarnings(self.warn)

    def _setup_tf_logging(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.tflog)

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
            logging.warning(f"Failed to set memory growth: {e}")

    def _configure_debug(self):
        if not self.tfdebug:
            return

        tf.debugging.set_log_device_placement(True)
        tf.config.run_functions_eagerly(True)
        tf.config.optimizer.set_jit(False)

        gpus = tf.config.list_physical_devices('GPU')
        logging.info(f"GPUs available: {gpus}")

        if gpus:
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                print("GPU Memory Info:", mem_info)
            except Exception as e:
                logging.warning(f"GPU memory info not available: {e}")

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
                logging.warning(f"{label} failed: {e}")

        raise RuntimeError("❌ No valid strategy available.")

    def set_log_dir(self, logdir=None, logfile='tslog', servername=None, backend=None):
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")

        if logdir is None:
            if hostname == servername and os_platform == 'Windows':
                base_path = r'C:\\WinRunMnt1\\8.0 Projects\\8.3 ProjectModelsEquinox\\EQUINRUN\\Logdir'
            elif os_platform == 'Linux':
                base_path = '/mnt/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            elif os_platform == 'Darwin':
                base_path = '/Users/shepa/OneDrive/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            else:
                base_path = os.path.expanduser('~/EQUINRUN/Logdir')
        else:
            base_path = logdir

        final_logdir = os.path.join(base_path, hostname, backend)
        os.makedirs(final_logdir, exist_ok=True)

        self.global_logdir = final_logdir
        # Ensure that the logfile name is based on the provided logfile argument
        # and has a .log extension.
        if not logfile.endswith('.log'):
            logfile = f"{logfile}.log"
        self.global_logfile = os.path.join(final_logdir, logfile)

        try:
            # Create the logfile if it doesn't exist, or just ensure writability
            with open(self.global_logfile, 'a', encoding='utf-8') as f: # Ensure UTF-8 when opening
                f.write('')
        except Exception as e:
            print(f"Could not create logfile at {self.global_logfile}: {e}")
            raise

        return self.global_logdir, self.global_logfile

    def setup_logging(self, **kwargs):
        """Sets up the logging configuration using RichHandler to log to a logfile and console."""
        logfile = kwargs.get('logfile', None)

        install(show_locals=True)  # Optional: Enhances Rich traceback for better debugging

        # Ensure logfile is provided or fallback to a default
        LOG_FILE = logfile or getattr(self, 'global_logfile', None)
        
        # Create a logger instance
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Set the minimum logging level

        # Remove any existing handlers to prevent duplicate logs if setup_logging is called multiple times
        if logger.hasHandlers():
            for handler in list(logger.handlers): # Iterate over a copy to safely remove
                logger.removeHandler(handler)

        # Create a file handler
        # IMPORTANT: Specify encoding='utf-8' for the FileHandler
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create a RichHandler for console output
        # IMPORTANT: Explicitly set `console=sys.stdout` and `encoding='utf-8'` for RichHandler's stream
        console_handler = RichHandler(
            level=logging.INFO,
            show_time=True,
            show_level=True,
            rich_tracebacks=True,
            console=sys.stdout, # Explicitly tell RichHandler to use stdout
            log_time_format="[%m/%d/%y %H:%M:%S]" # Optional: consistent time format
        )
        # Manually ensure the console stream is opened with utf-8 if not already
        if sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
                sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
            except Exception as e:
                # This re-opening might not always work or be necessary, log if it fails.
                logging.warning(f"Failed to reconfigure sys.stdout/stderr to UTF-8: {e}")

        # Create a formatter for the file handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info(f"Logging setup complete. Messages will be logged to {LOG_FILE}")
