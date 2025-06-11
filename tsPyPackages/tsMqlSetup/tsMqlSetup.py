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
                logger.info("GPU Memory Info:", mem_info)
            except Exception as e:
                logging.warning(f"GPU memory info not available: {e}")

        import psutil
        logger.info("RAM Used:", psutil.virtual_memory().used / 1e9, "GB")

        tf.keras.backend.clear_session()
        gc.collect()

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            logger.info("Using TPU")
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
                logger.info(f" Using {label}")
                return strategy
            except Exception as e:
                logging.warning(f"{label} failed: {e}")

        raise RuntimeError(" No valid strategy available.")

    def set_log_dir(self, logdir=None, logfile='tslog', servername=None, backend=None):
        hostname = socket.gethostname()
        # logger.info(f"Hostname: {hostname}") # This logger.info can be problematic before logging is fully set up

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
            # Use logger.info() here as logging might not be fully configured yet
            logger.info(f"ERROR: Could not create logfile at {self.global_logfile}: {e}")
            raise

        return self.global_logdir, self.global_logfile

    def setup_logging(self, **kwargs):
        """Sets up the logging configuration using RichHandler, loguru, and colorlog."""

        import sys
        import logging
        import os
        from loguru import logger as loguru_logger
        import colorlog
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.traceback import install

        logfile = kwargs.get('logfile', None)

        # Ensure Windows terminal supports UTF-8
        if sys.platform.startswith('win'):
            try:
                os.system('chcp 65001 > null')# Set console to UTF-8 encoding
                
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'    
            except Exception as e:
                print(f"WARNING: Failed to set Windows console encoding: {e}")

        # Install rich traceback
        install(show_locals=True)

        # Determine final logfile path
        final_logfile_path = logfile or self.global_logfile
        if not final_logfile_path:
            default_log_dir = Path(os.getcwd()) / "default_logs"
            default_log_dir.mkdir(parents=True, exist_ok=True)
            final_logfile_path = str(default_log_dir / "default_app.log")
            print(f"WARNING: Defaulting log file path to: {final_logfile_path}")

        # --- Setup colorlog as backup standard logging handler ---
        root_logger = logging.getLogger()
        root_logger.setLevel(self.loglevel)

        # Remove all existing logging handlers
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

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
        loguru_logger.remove()

        # File sink
        loguru_logger.add(
        final_logfile_path,
        level=self.loglevel.upper(),  # ✅ correct: returns 'DEBUG'
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        encoding="utf-8",
        enqueue=True
        )

        # Rich-compatible console sink
        console = Console(width=120)
        loguru_logger.add(lambda msg: console.print(msg, end=""), level=self.loglevel.upper())


        # --- Redirect standard logging to loguru ---
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = loguru_logger.level(record.levelname).name
                except Exception:
                    level = record.levelno
                loguru_logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

        logging.basicConfig(handlers=[InterceptHandler()], level=self.loglevel)

        # Register a fallback error handler for codecs if needed
        import codecs
        codecs.register_error('strict', codecs.ignore_errors)

        loguru_logger.info(f"Logging initialized. Logfile: {final_logfile_path}")
