import os
import warnings
import gc
import logging
import tensorflow as tf
import intel_tensorflow as itex
from tensorflow.keras.mixed_precision import Policy

from tsMqlPlatform import run_platform, platform_checker

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

logger = logging.getLogger(__name__)
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', precision='mixed_float16', tfdebug=False, num_cores=24, num_threads=2, **kwargs):
        self.tflog = tflog
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores = num_cores
        self.num_threads = num_threads
        self.sumthreads = self.num_cores * self.num_threads
        self.kwargs = kwargs

        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        logger.info(f"TF_CPP_MIN_LOG_LEVEL: {os.environ['TF_CPP_MIN_LOG_LEVEL']}")
        logger.info(f"TF_ENABLE_ONEDNN_OPTS: {os.environ['TF_ENABLE_ONEDNN_OPTS']}")

        print(f"TensorFlow Version: {tf.__version__}")
        tf.keras.mixed_precision.set_global_policy(Policy(self.precision))

        self.__set_gpu_memory_growth()
        self.__set_setup_tfdebug()
        self.__set_multi_threading()

    def __set_multi_threading(self):
        os.environ.update({
            "OMP_NUM_THREADS": str(self.num_cores),
            "TF_NUM_INTRAOP_THREADS": str(self.num_cores),
            "TF_NUM_INTEROP_THREADS": str(self.num_cores),
            "MKL_NUM_THREADS": str(self.num_cores),
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
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options({
            "auto_mixed_precision": True,
            "layout_optimizer": True,
            "mkl": True,
            "onednn": True
        })

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
            tf.config.run_functions_eagerly(True)
            tf.config.optimizer.set_jit(True)

            gpus = tf.config.list_physical_devices('GPU')
            logger.info(f"GPUs Available: {gpus}")

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if gpus:
                try:
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
        except ValueError:
            print("⚠️ TPU not found, using default strategy")
            return tf.distribute.get_strategy()

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Running on TPU")
            return tf.distribute.TPUStrategy(tpu)
        except (ValueError, tf.errors.NotFoundError) as e:
            print(f"⚠️ TPU not found: {e}")

        try:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print("✅ Running on MultiWorker GPU/CPU")
            return strategy
        except (tf.errors.InternalError, tf.errors.UnavailableError) as e:
            print(f"⚠️ MultiWorker strategy failed: {e}")

        try:
            strategy = tf.distribute.MirroredStrategy()
            print("✅ Running on Mirrored GPU/CPU")
            return strategy
        except (tf.errors.InternalError, tf.errors.UnavailableError) as e:
            print(f"⚠️ Mirrored strategy failed: {e}")

        try:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            print("✅ Running on CPU")
            return strategy
        except Exception as e:
            print(f"❌ Failed to initialize any strategy: {e}")
            raise RuntimeError("No valid computation strategy could be initialized.")

    def set_log_dir(self, logdir=None, logfile='tslog', servername=None):
        import socket
        hostname = os.getenv('HOSTNAME', socket.gethostname())
        print(f"Set log: Hostname: {hostname}")

        if logdir is None:
            if hostname == servername and os_platform == 'Windows':
                base_path = r'C:\\WinRunMnt1\\8.0 Projects\\8.3 ProjectModelsEquinox\\EQUINRUN\\Logdir'
            elif os_platform == 'Linux':
                base_path = '/mnt/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            elif os_platform == 'Darwin':
                base_path = '/Users/shepa/OneDrive/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir'
            else:
                base_path = r'C:\\Users\\shepa\\OneDrive\\8.0 Projects\\8.3 ProjectModelsEquinox\\EQUINRUN\\Logdir'

            os.makedirs(base_path, exist_ok=True)
            self.global_logdir = base_path
            self.global_logfile = os.path.join(base_path, logfile)
            if not os.path.exists(self.global_logfile):
                with open(self.global_logfile, 'w') as f:
                    f.write("Log file created successfully.")

        return self.global_logdir, self.global_logfile

    def set_logger(self, global_logfile):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        try:
            fh = logging.FileHandler(global_logfile, mode='w', encoding='utf-8')
        except OSError as e:
            print(f"Error creating log file: {e}")
            fh = logging.FileHandler('fallback.log', mode='w', encoding='utf-8')
            print("Fallback log file created: fallback.log")

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging configured successfully.")
        logger.info("Logfile: %s", global_logfile)
        return logger
