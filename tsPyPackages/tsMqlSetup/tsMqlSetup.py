import os
import warnings
import gc
import logging
import socket
import codecs
import io
import sys

# Set environment variables for TensorFlow optimizations
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tsMqlPlatform import run_platform, platform_checker
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path # Import Path for directory handling

# Initialize platform checkers
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

class CMqlSetup:
    def __init__(self, loglevel='DEBUG', tflog=2, warn='ignore', precision='mixed_float16', tfdebug=False, num_cores=48, num_threads=8, **kwargs):
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

        # Initialize global logfile and logdir paths as None, they will be set by set_log_dir
        self.global_logdir = None
        self.global_logfile = None

        self._setup_warnings()
        self._setup_tf_logging()
        self._set_precision_policy()
        self._configure_tf()
        self._configure_debug()

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
            print("Using TPU")
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
                print(f" Using {label}")
                return strategy
            except Exception as e:
                logging.warning(f"{label} failed: {e}")

        raise RuntimeError(" No valid strategy available.")

    def set_log_dir(self, logdir=None, logfile='tslog', servername=None, backend=None):
        hostname = socket.gethostname()
        # print(f"Hostname: {hostname}") # This print can be problematic before logging is fully set up

        if logdir is None:
            # Determine base path dynamically
            # Assuming a structure where the script is run from project_root/PythonLib/tsProjects/...
            # and project_root is the base for Logdir, MQL5 etc.
            script_dir = Path(__file__).parent
            # Navigate up to the project root (e.g., EQUINRUN)
            # Adjust this logic if your directory structure differs
            project_root = script_dir
            for _ in range(5): # Go up 5 levels to reach EQUINRUN, adjust as needed
                if project_root.name == 'EQUINRUN':
                    break
                project_root = project_root.parent
            else: # If loop finishes without finding 'EQUINRUN'
                project_root = Path(os.getcwd()) # Fallback to current working directory

            # Construct default logdir based on project_root
            if os_platform == 'Windows':
                base_path = project_root / 'Logdir'
            elif os_platform == 'Linux':
                base_path = project_root / 'Logdir'
            elif os_platform == 'Darwin':
                base_path = project_root / 'Logdir'
            else:
                base_path = Path(os.path.expanduser('~')) / 'EQUINRUN' / 'Logdir'
            
            base_path = str(base_path) # Convert Path object to string for os.path.join compatibility
        else:
            base_path = logdir

        # Ensure hostname and backend are valid strings for path
        hostname_str = str(servername) if servername else hostname
        backend_str = str(backend) if backend else 'unknown_backend'

        final_logdir = os.path.join(base_path, hostname_str, backend_str)
        os.makedirs(final_logdir, exist_ok=True)

        self.global_logdir = final_logdir
        # Ensure that the logfile name is based on the provided logfile argument
        # and has a .log extension.
        if not logfile.endswith('.log'):
            logfile = f"{logfile}.log"
        self.global_logfile = os.path.join(final_logdir, logfile)

        try:
            # Create the logfile if it doesn't exist, or just ensure writability
            # Using 'x' mode to create if not exists, avoid overwriting existing.
            # If it exists, 'a' mode (append) is fine.
            with open(self.global_logfile, 'a', encoding='utf-8') as f:
                f.write('') # Just touch the file to ensure it's writable
        except Exception as e:
            # Use print() here as logging might not be fully configured yet
            print(f"ERROR: Could not create logfile at {self.global_logfile}: {e}")
            raise

        return self.global_logdir, self.global_logfile

    def setup_logging(self, **kwargs):
        """Sets up the logging configuration using RichHandler to log to a logfile and console."""
        logfile = kwargs.get('logfile', None)

        # CRITICAL: Configure console encoding for Windows before any Rich initialization
        if sys.platform.startswith('win'):
            try:
                # Set the console's code page to UTF-8. `> nul` suppresses its output.
                # This should be executed via os.system for the current cmd window.
                os.system('chcp 65001 > nul')
                # Set environment variables, crucial for child processes to inherit UTF-8
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
            except Exception as e:
                # Use print() here as logging might not be fully configured yet
                print(f"WARNING: Failed to set console encoding to UTF-8: {e}")

        # Ensure Rich's traceback handler is installed early
        install(show_locals=True)

        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(self.loglevel) # Use the loglevel from CMqlSetup instance

        # Remove any existing handlers to prevent duplicate logs if setup_logging is called multiple times
        # This is important if this function is called more than once.
        if logger.hasHandlers():
            for handler in list(logger.handlers): # Iterate over a copy to safely remove
                logger.removeHandler(handler)

        # Determine the log file path. Prioritize kwargs, then instance attribute.
        final_logfile_path = logfile
        if not final_logfile_path and self.global_logfile:
            final_logfile_path = self.global_logfile
        
        # Fallback if no specific logfile path is provided even after self.global_logfile check
        if not final_logfile_path:
            # A very basic fallback path, primarily for dev/debugging if set_log_dir isn't called first
            default_log_dir = Path(os.getcwd()) / "default_logs"
            default_log_dir.mkdir(parents=True, exist_ok=True)
            final_logfile_path = str(default_log_dir / "default_app.log")
            print(f"WARNING: No specific logfile path provided. Defaulting to: {final_logfile_path}")


        # Create a file handler, always specifying UTF-8 encoding
        try:
            file_handler = logging.FileHandler(final_logfile_path, encoding='utf-8')
            file_handler.setLevel(self.loglevel)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"ERROR: Failed to set up file logging to {final_logfile_path}: {e}")

        # Create a RichHandler for console output
        # Rich will use the `sys.stdout` stream, which we've attempted to configure for UTF-8
        console_handler = RichHandler(
            level=self.loglevel,
            show_time=True,
            show_level=True,
            rich_tracebacks=True,
            log_time_format="[%m/%d/%y %H:%M:%S]" # Consistent time format
        )
        logger.addHandler(console_handler)

        # Register a lenient error handler for codecs, in case of lingering issues
        codecs.register_error('strict', codecs.ignore_errors)

        logging.info(f"Logging setup complete. Messages will be logged to {final_logfile_path}")
