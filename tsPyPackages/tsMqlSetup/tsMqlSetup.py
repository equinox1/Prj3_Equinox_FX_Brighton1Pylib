import os
import warnings
import gc
import logging
import socket
import codecs
import io
import sys
from loguru import logger as loguru_logger
import colorlog

# Set environment variables for TensorFlow optimizations
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tsMqlPlatform import run_platform, platform_checker
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path

# Initialize platform checkers
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

class CMqlSetup:
    _log_setup_done = False # Class-level flag to track if initial logging is done

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

        # Initialize global logfile and logdir paths as None; they will be set by set_log_dir
        self.global_logdir = None
        self.global_logfile = None

        # These setups should be safe to call multiple times or are idempotent
        self._setup_warnings()
        self._setup_tf_logging()
        self._set_precision_policy()
        self._configure_tf()
        # _configure_debug might try to log, so ensure main logger is set up first by launcher
        # self._configure_debug() 

    def _setup_warnings(self):
        warnings.filterwarnings(self.warn)

    def _setup_tf_logging(self):
        # This only sets an environment variable, which is idempotent
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.tflog)

    def _set_precision_policy(self):
        # This sets a global TensorFlow policy, which is idempotent
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))

    def _configure_tf(self):
        # Environment variable updates and TensorFlow config are idempotent
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
                logging.info(f"Enabled memory growth for GPU: {gpu}")
        except RuntimeError as e:
            # Use print() here, as logging might not be fully configured yet
            print(f"WARNING: Failed to set memory growth for GPU: {e}", file=sys.stderr)
            logging.warning(f"Failed to set memory growth for GPU: {e}")
            

    def _configure_debug(self):
        # This method is called conditionally and can contain logging.
        # It's safer to only call this AFTER the main logging is set up
        # or handle its output carefully (e.g., using print for early messages).
        if not self.tfdebug:
            return

        tf.debugging.set_log_device_placement(True)
        tf.config.run_functions_eagerly(True)
        tf.config.optimizer.set_jit(False)

        gpus = tf.config.list_physical_devices('GPU')
        logging.info(f"GPUs available: {gpus}") # This logging.info will use the configured root logger

        if gpus:
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                logging.info(f"GPU Memory Info: {mem_info}")
            except Exception as e:
                logging.warning(f"GPU memory info not available: {e}")

        import psutil
        logging.info(f"RAM Used: {psutil.virtual_memory().used / 1e9:.2f} GB")

        tf.keras.backend.clear_session()
        gc.collect()

    def get_computation_strategy(self):
        """Determines and returns the appropriate TensorFlow distribution strategy."""
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            logging.info("Using TPU")
            return tf.distribute.TPUStrategy(tpu)
        except Exception:
            pass # TPU not available or failed to connect

        for strategy_cls, label in [
            (tf.distribute.MultiWorkerMirroredStrategy, "MultiWorker GPU/CPU"),
            (tf.distribute.MirroredStrategy, "Mirrored GPU/CPU"),
            (lambda: tf.distribute.OneDeviceStrategy("/cpu:0"), "CPU (OneDevice)"),
            (tf.distribute.experimental.ParameterServerStrategy, "Parameter Server"),
            (tf.distribute.experimental.CentralStorageStrategy, "Central Storage"),
        ]:
            try:
                strategy = strategy_cls()
                logging.info(f" Using {label}")
                return strategy
            except Exception as e:
                logging.warning(f"{label} failed: {e}")

        raise RuntimeError("No valid strategy available.")

    def set_log_dir(self, logdir=None, logfile='tslog.log', servername=None, backend=None):
        """
        Determines and creates the logging directory and file path.
        Sets self.global_logdir and self.global_logfile.
        """
        hostname = socket.gethostname()

        if logdir is None:
            # Determine base path dynamically
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir
            # Navigate up to the project root (e.g., EQUINRUN)
            # Adjust this range/logic if your directory structure differs
            for _ in range(5): # Go up to 5 levels to find 'EQUINRUN'
                if project_root.name == 'EQUINRUN':
                    break
                project_root = project_root.parent
            else: # If loop finishes without finding 'EQUINRUN'
                project_root = Path(os.getcwd()) # Fallback to current working directory

            # Construct default logdir based on project_root
            # Ensure base_path is a Path object for consistent operations
            base_path = project_root / 'Logdir'
            
        else:
            base_path = Path(logdir) # Ensure it's a Path object if provided

        # Ensure hostname and backend are valid strings for path construction
        hostname_str = str(servername) if servername else hostname
        backend_str = str(backend) if backend else 'unknown_backend'

        # Construct the final log directory path
        final_logdir = base_path / hostname_str / backend_str
        final_logdir.mkdir(parents=True, exist_ok=True) # Create directories if they don't exist

        self.global_logdir = str(final_logdir) # Store as string
        
        # Ensure logfile name has a .log extension
        if not logfile.endswith('.log'):
            logfile = f"{logfile}.log"
        self.global_logfile = str(final_logdir / logfile) # Store as string

        try:
            # Touch the file to ensure it's created and writable.
            # Using 'a' mode ensures it's created if it doesn't exist, and doesn't truncate.
            with open(self.global_logfile, 'a', encoding='utf-8') as f:
                f.write('') # Just touch the file
        except Exception as e:
            # Use print to report critical errors before full logging is guaranteed to be up
            print(f"CRITICAL ERROR: Could not create logfile at {self.global_logfile}: {e}", file=sys.stderr)
            raise

        return self.global_logdir, self.global_logfile

    def setup_logging(self, **kwargs):
        """
        Sets up the logging configuration using RichHandler, loguru, and colorlog.
        This method is designed to be called once per process.
        """
        # Use a class-level flag to ensure setup runs only once per process
        if CMqlSetup._log_setup_done:
            return

        logfile = kwargs.get('logfile', None)

        # Ensure Windows terminal supports UTF-8
        if sys.platform.startswith('win'):
            try:
                os.system('chcp 65001 > null') # Set console to UTF-8 encoding
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'    
            except Exception as e:
                # Use print as logging might not be fully configured yet
                print(f"WARNING: Failed to set Windows console encoding: {e}", file=sys.stderr)

        # Install rich traceback, safe to call multiple times but usually done once
        install(show_locals=True)

        # Determine final logfile path, falling back to instance variable or default
        final_logfile_path = logfile or self.global_logfile
        if not final_logfile_path:
            # This fallback should ideally not be hit if set_log_dir is called first.
            default_log_dir = Path(os.getcwd()) / "default_logs"
            default_log_dir.mkdir(parents=True, exist_ok=True)
            final_logfile_path = str(default_log_dir / "default_app.log")
            # Use print for this warning as logger might not be fully configured
            print(f"WARNING: Defaulting log file path to: {final_logfile_path} (setup_logging fallback)", file=sys.stderr)

        # Define InterceptHandler class *before* it's used in the 'any' check
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # Get the correct loguru level name or use levelno fallback
                try:
                    level = loguru_logger.level(record.levelname).name
                except ValueError: # loguru_logger.level raises ValueError if level name is not found
                    level = record.levelno
                # Use opt(depth=...) to correct the stack trace depth in loguru
                loguru_logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


        # --- Setup colorlog for standard logging (fallback/console) ---
        root_logger = logging.getLogger()
        root_logger.setLevel(self.loglevel)

        # Remove existing standard StreamHandlers to prevent duplicates if this is called multiple times
        # in contexts where root_logger.handlers might accumulate.
        # This is less aggressive than removing *all* handlers, allowing file handlers to persist.
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, colorlog.StreamHandler):
                root_logger.removeHandler(handler)
        
        # Add colorlog StreamHandler only if one doesn't exist
        if not any(isinstance(h, colorlog.StreamHandler) for h in root_logger.handlers):
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            color_handler = colorlog.StreamHandler()
            color_handler.setFormatter(color_formatter)
            color_handler.setLevel(self.loglevel)
            root_logger.addHandler(color_handler)


        # --- Setup loguru ---
        # Remove existing loguru handlers. loguru.remove() is idempotent.
        loguru_logger.remove()

        # Add file sink for loguru ONLY IF a file handler for this path isn't already there
        # This is tricky because loguru manages its own sinks, not standard handlers.
        # The most reliable way for loguru to act as a singleton is to explicitly manage its sinks.
        # Given we redirect standard logging to loguru, the main file logging should be via loguru.
        # So, we always add the loguru file sink here if it's the central setup point.
        loguru_logger.add(
            final_logfile_path,
            level=self.loglevel.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", # Added padding for level
            encoding="utf-8",
            enqueue=True, # Use multiprocessing-safe queue
            rotation="10 MB", # Rotate log file at 10 MB
            compression="zip", # Compress old log files
            retention="7 days" # Keep logs for 7 days
        )

        # Rich-compatible console sink for loguru
        # Add this only if not already present or if we want to ensure it's the primary console output
        if not any(isinstance(sink, dict) and sink.get('handler') == Console for sink in loguru_logger._core.handlers.values()):
            console = Console(width=120)
            loguru_logger.add(lambda msg: console.print(msg, end=""), level=self.loglevel.upper(), colorize=True)


        # --- Redirect standard logging to loguru ---
        # This InterceptHandler ensures all standard logging calls (from other modules)
        # go through loguru. This should also only be set up once.
        if not any(isinstance(h, InterceptHandler) for h in root_logger.handlers): # Now InterceptHandler is defined
            # Remove any previous InterceptHandlers if they exist, then add this one
            for handler in root_logger.handlers[:]:
                if isinstance(handler, InterceptHandler):
                    root_logger.removeHandler(handler)
            root_logger.addHandler(InterceptHandler())

        # Ensure basicConfig is not used in a way that interferes with existing handlers
        # It's better to manage handlers manually as done above.
        # logging.basicConfig(handlers=[InterceptHandler()], level=self.loglevel) # This line should be removed or commented out if InterceptHandler is added manually.

        # Register a fallback error handler for codecs if needed
        codecs.register_error('strict', codecs.ignore_errors)

        # Mark logging setup as complete for this process
        CMqlSetup._log_setup_done = True
        loguru_logger.info(f"Logging initialized. Logfile: {final_logfile_path}")

