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
        self._encoding()

    def _encoding(self):
        if sys.platform.startswith('win'):
            if sys.getfilesystemencoding() != 'utf-8':
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
                codecs.register_error('strict', codecs.ignore_errors)
            
                os.system('chcp 65001')  # Set UTF-8 codepage in console
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')


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

    def set_log_dir(self, logdir=None, logfile='tslog', servername=None, ltuner=None):
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

    def setup_global_logger(self, logfile_path, force_reset=False):
        """
        Sets up a global logger with console output (via RichHandler) and file output.
        Includes robust handling for rewrapping sys.stdout/sys.stderr on Windows.

        Args:
            logfile_path (str): The path to the log file.
            force_reset (bool): If True, the log file will be overwritten; otherwise,
                                logs will be appended.

        Returns:
            logging.Logger: The configured logger instance.
        """
        import threading

        # --- Safe rewrap only if we're in the main thread on Windows ---
        if sys.platform.startswith('win') and threading.current_thread() is threading.main_thread():
            # Attempt to rewrap sys.stdout
            # DEBUG:print(f"sys.stderr={sys.stderr}, type={type(sys.stderr)}, closed={getattr(sys.stderr, 'closed', 'N/A')}", file=sys.__stdout__)

            try:
                if (
                    sys.stdout and
                    hasattr(sys.stdout, 'buffer') and
                    not getattr(sys.stdout, 'closed', False) and
                    not isinstance(sys.stdout, io.TextIOWrapper)
                ):
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"Warning: Failed to rewrap sys.stdout: {e}", file=sys.__stdout__)

            # Attempt to rewrap sys.stderr
            try:
                if (
                    sys.stderr and
                    hasattr(sys.stderr, 'buffer') and
                    not getattr(sys.stderr, 'closed', False) and
                    not isinstance(sys.stderr, io.TextIOWrapper)
                ):
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"Warning: Failed to rewrap sys.stderr: {e}", file=sys.__stderr__ if sys.__stderr__ else sys.__stdout__)

        # --- Proceed with standard logger setup ---
        loglevel = getattr(logging, self.loglevel.upper(), logging.INFO)

        handlers = [RichHandler(rich_tracebacks=True)]

        try:
            file_handler = logging.FileHandler(logfile_path, mode='w' if force_reset else 'a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'))
            handlers.append(file_handler)
        except Exception as e:
            print(f"❌ Failed to initialize file logging: {e}", file=sys.__stderr__ if sys.__stderr__ else sys.__stdout__)
            raise

        logging.basicConfig(level=loglevel, handlers=handlers, force=True)

        logger = logging.getLogger()
        logger.info(f"Logger initialized with file: {logfile_path}")
        return logger
