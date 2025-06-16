import os
import warnings
import gc
import logging
import socket
import codecs
import io
import sys
from loguru import logger as loguru_logger
import inspect

from tsMqlPlatform import run_platform, platform_checker
from rich.console import Console # Keep Console for potential direct use, but simplify add()
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path

# Initialize platform checkers - these are global to the module
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

class CMqlLogService:
    """
    Manages centralized logging configuration for the application using both
    standard Python logging and Loguru for rich output and advanced features.
    """

    def __init__(self, **kwargs):
        """
        Initializes the logging service.

        Args:
            loglevel (str): The desired logging level (e.g., 'INFO', 'DEBUG').
                            Can be passed via kwargs.
            logdir (str | Path, optional): The base directory for log files. If None, it
                                          will try to derive it from environment variables
                                          or project structure. Can be passed via kwargs.
            logfile (str, optional): The name of the log file. Defaults to 'tslog.log'.
                                     Can be passed via kwargs.
            servername (str, optional): The name of the server, used in log directory path.
                                        Can be passed via kwargs.
            backend (str, optional): The ML backend (e.g., 'tensorflow', 'pytorch'), used
                                     in log directory path. Can be passed via kwargs.
            enable_logging (bool): If False, logging will be effectively disabled.
                                   Defaults to True. Can be passed via kwargs.
        """
        self.kwargs = kwargs
        self.global_logdir = None
        self.global_logfile = None
        self.enable_logging = kwargs.get('enable_logging', True)
        self.loglevel = kwargs.get('loglevel', "INFO").upper()

        if self.enable_logging:
            # We need app_params and base_params early to determine log paths
            # Defer import to prevent circular dependency if CMqlOverrides itself needs logging
            # during its import/init, although in practice it's less common for core config
            # to depend on the logging service being fully initialized.
            from tsMqlOverrides import CMqlOverrides
            mql_overrides = CMqlOverrides()
            all_params = mql_overrides.env.all_params()
            self.app_params = all_params.get("app", {})
            self.tune_params = all_params.get("mltune", {})
            self.base_params = all_params.get('base', {})

            # Use kwargs values if provided, otherwise fall back to app_params/base_params
            _logdir = kwargs.get('logdir', self.base_params.get('mp_glob_base_log_path'))
            _logfile = kwargs.get('logfile', self.app_params.get('xerces_logfile', 'tslog.log'))
            _servername = kwargs.get('servername', self.app_params.get('xerces_servername', 'localhost'))
            _backend = kwargs.get('backend', self.tune_params.get('backend', 'unknown_backend'))

            self._set_log_paths(_logdir, _logfile, _servername, _backend)
            self._configure_debug() # Configure rich traceback etc.
        else:
            # If logging is disabled, ensure no file handlers are set up
            logging.disable(logging.CRITICAL) # Disable all logging from standard logger
            loguru_logger.remove() # Remove all existing handlers from Loguru
            # Keep only critical messages to stderr if logging is disabled, with minimal format
            loguru_logger.add(sys.stderr, level="CRITICAL", format="{message}", colorize=True)
            print("Logging is disabled by configuration.", file=sys.stderr)

    def _set_log_paths(self, logdir_arg, logfile_arg, servername_arg, backend_arg):
        """
        Determines and sets the global log directory and file path.
        Ensures logs are stored in Logdir/<backend>/tsneuropredict_app.log
        """
        # Determine the base log directory
        if logdir_arg:
            base_log_dir = Path(logdir_arg)
        elif os.environ.get("LOGDIR"):
            base_log_dir = Path(os.environ["LOGDIR"])
        else:
            # Fallback: derive Logdir from script's location
            script_dir = Path(__file__).resolve().parent
            base_log_dir = script_dir.parent / "Logdir"

        # Determine backend name for subfolder
        backend_str = os.environ.get('BACKEND', self.tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified


        # Final log directory: Logdir/<backend>
        print(f"LOGSERVICE:Setting up logging in directory: {base_log_dir} backend_str {backend_str}", file=sys.stderr)
        final_logdir = base_log_dir / backend_str
        try:
            final_logdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"CRITICAL ERROR: Could not create log directory: {final_logdir}: {e}", file=sys.stderr)
            raise RuntimeError(f"Failed to create log directory: {final_logdir}") from e

        self.global_logdir = str(final_logdir)

        # Determine log filename (default to tsneuropredict_app.log)
        if logfile_arg:
            logfilename = Path(logfile_arg).name
            if not logfilename.endswith(".log"):
                logfilename += ".log"
        else:
            logfilename = "tsneuropredict_app.log"

        self.global_logfile = str(final_logdir / logfilename)
        print(f"LOGSERVICE:Setting up logging in file: {self.global_logfile}", file=sys.stderr)

        # Test write permissions
        try:
            with open(self.global_logfile, 'a', encoding='utf-8') as f:
                f.write('')
        except Exception as e:
            print(f"CRITICAL ERROR: Could not access logfile: {self.global_logfile}: {e}", file=sys.stderr)
            raise RuntimeError(f"Failed to initialize logfile: {self.global_logfile}") from e


    def _configure_debug(self):
        """Configures Rich for traceback and console output."""
        # Correctly pass the parent directory as a string for suppression
        # Suppress frames from tsMqlLogService itself for cleaner tracebacks
        install(show_locals=True, suppress=[str(Path(__file__).parent)])

    def setup_logging(self, **kwargs):
        """
        Sets up the logging configuration for both the standard logging module and Loguru.
        This method will do nothing if self.enable_logging is False.
        """
        if not self.enable_logging:
            print("Logging is disabled, skipping setup_logging.", file=sys.stderr)
            return

        final_logfile_path = kwargs.get('logfile', self.global_logfile)
        if not final_logfile_path:
            # This should ideally not happen if _set_log_paths was successful
            raise RuntimeError("Logging path not set. Call `_set_log_paths()` or provide `logfile`.")

        # Configure console encoding for Windows if not already set
        if sys.platform.startswith('win'):
            # Only set if not already UTF-8, to avoid potential issues with re-opening streams
            if sys.stdout.encoding.lower() != 'utf-8':
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                # For immediate effect, you might need to re-configure streams,
                # but it's often better to rely on env var for subprocesses
                # and ensure the terminal itself is UTF-8 capable.
                print("INFO: Set PYTHONIOENCODING to UTF-8 for Windows console.", file=sys.stderr)

        # Configure standard logging to use Loguru via an intercept handler
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    # Map standard logging levels to Loguru levels
                    level = loguru_logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                # Route standard logging records to Loguru
                # depth=6 ensures correct source file/line info when logging via standard logger
                loguru_logger.opt(depth=6, exception=record.exc_info, raw=False).log(level, record.getMessage())

        root_logger = logging.getLogger()
        root_logger.setLevel(self.loglevel)

        # Remove all existing handlers from the root logger to prevent duplicate output
        # This is critical for re-initialization scenarios or preventing default handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add the InterceptHandler to the root logger
        root_logger.addHandler(InterceptHandler())

        # Configure Loguru
        loguru_logger.remove() # Remove default Loguru handler (stdout) to start fresh

        # Add file sink for Loguru
        loguru_logger.add(
            final_logfile_path,
            level=self.loglevel,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8",
            enqueue=True, # Use multiprocessing-safe queue
            rotation="10 MB",
            compression="zip",
            retention="7 days",
            diagnose=True # Include debugging information in logs for file
        )

        # Add console sink for Loguru with rich styling
        try:
            loguru_logger.add(
                sys.stderr, # Using stderr for console output is common practice for logs
                level=self.loglevel,
                colorize=True,
                diagnose=True, # Enables rich tracebacks for Loguru's console output
                format="{time:HH:mm:ss} | {level: <8} | {message}" # Simpler console format
            )
        except Exception as e:
            print(f"WARNING: Could not set up Loguru console logging (might be missing Rich dependencies or unusual terminal config): {e}", file=sys.stderr)

        codecs.register_error('strict', codecs.ignore_errors)

        loguru_logger.info(f"Logging initialized successfully. Logfile: {final_logfile_path}")


# Import tensorflow here to avoid circular dependency with CMLogServiceSetup if TF is needed for CMLogServiceSetup init.
# Assuming TensorFlow is installed and available in the environment.
try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
except ImportError:
    tf = None
    mixed_precision = None
    # Use print for this warning as logging might not be fully setup yet
    print("WARNING: TensorFlow not found. Some features may be unavailable.", file=sys.stderr)


class CMLogServiceSetup(CMqlLogService): # Inherit from CMqlLogService for logging capabilities
    def __init__(self, loglevel='INFO', warn='ignore', precision='float32', tfdebug=False, num_cores=None, num_threads=None, enable_logging=True, **kwargs):
        """
        Initializes the CMLogServiceSetup class, configuring environment and logging.

        Args:
            loglevel (str): Logging level.
            warn (str): Warning filter action.
            precision (str): TensorFlow mixed precision policy.
            tfdebug (bool): TensorFlow debug flag.
            num_cores (int, optional): Number of CPU cores to use.
            num_threads (int, optional): Number of CPU threads to use.
            enable_logging (bool): If False, logging will be effectively disabled. Defaults to True.
            **kwargs: Additional keyword arguments to pass to CMqlLogService.__init__,
                      e.g., `logdir`, `logfile`, `servername`, `backend`.
        """
        # Load parameters from tsMqlOverrides early to pass to parent constructor
        # This prevents redundant calls to all_params() and ensures consistency.
        from tsMqlOverrides import CMqlOverrides
        mql_overrides = CMqlOverrides()
        all_params = mql_overrides.env.all_params()
        app_params = all_params.get("app", {})
        base_params = all_params.get('base', {})
        tune_params = all_params.get('mltune', {})

        # Merge kwargs with config parameters, giving kwargs precedence
        _loglevel = kwargs.get('loglevel', loglevel).upper()
        _logdir = kwargs.get('logdir', base_params.get('mp_glob_base_log_path'))
        _logfile = kwargs.get('logfile', app_params.get('xerces_logfile', 'tslog.log'))
        _servername = kwargs.get('servername', app_params.get('xerces_servername', 'localhost'))
        _backend = kwargs.get('backend', tune_params.get('backend', 'unknown_backend'))
        _enable_logging = kwargs.get('enable_logging', enable_logging)

        # Call the parent CMqlLogService's constructor to set up logging paths and debug
        super().__init__(
            loglevel=_loglevel,
            logdir=_logdir,
            logfile=_logfile,
            servername=_servername,
            backend=_backend,
            enable_logging=_enable_logging
        )

        self.loglevel = _loglevel # Store final loglevel
        self.warn = warn
        self.precision = precision
        self.tfdebug = tfdebug
        self.num_cores = num_cores
        self.num_threads = num_threads

        if self.enable_logging: # Only apply global settings if logging is enabled
            self._apply_global_settings()
            # Use loguru_logger here as it's now configured globally via CMqlLogService.__init__
            loguru_logger.info("CMLogServiceSetup initialized and global settings applied.")
        else:
            print("CMLogServiceSetup initialized with logging disabled.", file=sys.stderr)


    def _apply_global_settings(self):
        """Applies global settings for warnings and TensorFlow."""
        warnings.filterwarnings(self.warn)
        # Configure TensorFlow
        if tf: # Check if TensorFlow was imported successfully
            if tf.config.list_physical_devices('GPU'):
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
                loguru_logger.info("GPU found and memory growth set to True.")
            else:
                loguru_logger.info("No GPU devices found.")

            # Set mixed precision policy
            try:
                policy = mixed_precision.Policy(self.precision)
                mixed_precision.set_global_policy(policy)
                loguru_logger.info(f"TensorFlow global mixed precision policy set to: {policy.name}")
            except Exception as e:
                loguru_logger.warning(f"Failed to set mixed precision policy '{self.precision}': {e}")

            # Configure CPU parallelism
            # Check if set_inter_op_parallelism_threads and set_intra_op_parallelism_threads exist
            # as they might vary with TF versions
            if self.num_cores and hasattr(tf.config.threading, 'set_inter_op_parallelism_threads'):
                tf.config.threading.set_inter_op_parallelism_threads(self.num_cores)
                loguru_logger.info(f"TensorFlow inter-op parallelism threads set to: {self.num_cores}")
            if self.num_threads and hasattr(tf.config.threading, 'set_intra_op_parallelism_threads'):
                tf.config.threading.set_intra_op_parallelism_threads(self.num_threads)
                loguru_logger.info(f"TensorFlow intra-op parallelism threads set to: {self.num_threads}")

            if self.tfdebug:
                if hasattr(tf.data.experimental, 'enable_debug_mode'):
                    tf.data.experimental.enable_debug_mode()
                    loguru_logger.info("TensorFlow debug mode enabled.")
                else:
                    loguru_logger.warning("TensorFlow data experimental debug mode not available in this TF version.")
        else:
            loguru_logger.info("TensorFlow not available, skipping TF global settings.")

        # Other global settings
        gc.enable() # Enable garbage collector
        loguru_logger.info("Garbage collector enabled.")


    @staticmethod
    def initialize_logging(role_hint=None, **kwargs) -> logging.Logger:
        """
        Static method to initialize the logging system for the application.
        This should be called early in the application's lifecycle.

        Args:
            role_hint (str, optional): A string indicating the role of the calling script
                                       (e.g., 'chief', 'worker', 'oracle_server'). Used for
                                       naming log files and subdirectories, typically maps to backend.
            **kwargs: Additional keyword arguments to pass to CMqlLogService.__init__
                      and CMLogServiceSetup.__init__, e.g., `loglevel`, `enable_logging`,
                      `logdir`, `logfile`, `servername`, `backend`.

        Returns:
            logging.Logger: A standard Python logger instance for the calling module.
        """
        # Load parameters needed for logging setup from tsMqlOverrides
        from tsMqlOverrides import CMqlOverrides
        mql_overrides = CMqlOverrides()
        all_params = mql_overrides.env.all_params()
        app_params = all_params.get("app", {})
        base_params = all_params.get('base', {})
        mltune_params = all_params.get('mltune', {})

        # Determine loglevel, prioritizing kwargs > app_params > default
        loglevel_final = kwargs.get('loglevel', app_params.get('loglevel', 'INFO'))
        # Determine if logging should be enabled, prioritizing kwargs > default
        enable_logging_final = kwargs.get('enable_logging', True)

        # Determine the base log directory, prioritizing kwargs > base_params > env var > derived
        central_logdir = kwargs.get('logdir', base_params.get('mp_glob_base_log_path'))
        if not central_logdir:
            central_logdir = os.environ.get('LOGDIR')

        if not central_logdir:
            # Fallback if no config or env var, attempt to find 'EQUINRUN'
            _script_dir = Path(inspect.currentframe().f_back.f_globals['__file__']).resolve().parent
            _project_root = _script_dir
            _found_equinrun = False
            for _ in range(5):
                if _project_root.name == 'EQUINRUN':
                    _found_equinrun = True
                    break
                if _project_root == _project_root.parent:
                    break
                _project_root = _project_root.parent

            if _found_equinrun:
                central_logdir = str(_project_root / 'Logdir')
            else:
                # If EQUINRUN not found, use a 'Logdir' relative to the script's parent
                central_logdir = str(_script_dir.parent / 'Logdir')
                print(f"WARNING: Could not find 'EQUINRUN' in path. Using '{central_logdir}' for base logs.", file=sys.stderr)

        # Ensure the central log directory exists if logging is enabled
        if enable_logging_final and central_logdir and not Path(central_logdir).is_dir():
            print(f"INFO: Central log directory '{central_logdir}' does not exist. Attempting to create.", file=sys.stderr)
            try:
                Path(central_logdir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"CRITICAL ERROR: Failed to create central log directory {central_logdir}: {e}. Logging may fail.", file=sys.stderr)
                # Do not raise here, allow CMqlLogService to handle subsequent failures gracefully
                pass

        # Determine dynamic parts of the log file path
        script_name = Path(sys.argv[0]).stem if len(sys.argv) > 0 else 'unknown_script'
        hostname = os.environ.get("COMPUTERNAME") or socket.gethostname()

        # Use role_hint for the backend part of the path and for logfile naming,
        # otherwise fall back to mltune_params['backend'] or a default.
        backend_for_path = kwargs.get('backend', role_hint or mltune_params.get('backend', 'unknown_backend_role'))

        # Determine the logfile name. Prioritize kwargs, then app_params, then dynamic.
        configured_logfile_name = kwargs.get('logfile', app_params.get('xerces_logfile'))
        final_logfile_base_name = configured_logfile_name # Start with configured name

        if not final_logfile_base_name:
            # Fallback to a fully dynamic name if no explicit logfile is configured
            final_logfile_base_name = f"{script_name}_{backend_for_path}_{hostname.lower()}.log"
        elif not str(final_logfile_base_name).endswith('.log'):
            final_logfile_base_name = f"{final_logfile_base_name}.log"

        # If a specific logfile is configured, we still might want to append role/hostname for distinction
        # especially in multi-instance deployments using the same base logfile name.
        # This specific logic (appending _backend_hostname) is already handled in _set_log_paths
        # if final_logfile_base_name is passed as `logfile_arg`.

        # Instantiate CMLogServiceSetup (which calls CMqlLogService's __init__)
        log_setup_instance = CMLogServiceSetup(
            loglevel=loglevel_final,
            logdir=str(central_logdir), # Ensure string for path
            logfile=final_logfile_base_name,
            servername=kwargs.get('servername', app_params.get('xerces_servername', hostname)),
            backend=backend_for_path,
            enable_logging=enable_logging_final,
            warn=kwargs.get('warn', 'ignore'),
            precision=kwargs.get('precision', 'float32'),
            tfdebug=kwargs.get('tfdebug', False),
            num_cores=kwargs.get('num_cores'),
            num_threads=kwargs.get('num_threads')
        )

        if enable_logging_final:
            # setup_logging is called implicitly via CMqlLogService.__init__ within CMLogServiceSetup.__init__
            # if enable_logging is True. No need to call it again explicitly here unless a re-initialization
            # with new parameters is desired, which isn't the primary goal of this static method.
            # However, for clarity, if the instance didn't call it, we could.
            # Given the current structure, super().__init__ will trigger it.
            pass

        # Return a standard Python logger for the module that called initialize_logging
        # This ensures correct source information (module name, line number) in logs.
        calling_frame = inspect.currentframe().f_back
        calling_module_name = calling_frame.f_globals.get('__name__', 'unknown_module')
        return logging.getLogger(calling_module_name)


# Example client usage (for demonstration/testing)
if __name__ == "__main__":
    print("--- Running tsMqlLogService.py as main module ---")

    # Simulate tsMqlOverrides for testing purposes
    # In a real setup, you would have a tsMqlOverrides.py file
    # For this test, we create a mock version
    class MockEnv:
        def all_params(self):
            return {
                "app": {
                    "xerces_logfile": "tsneuropredict_app.log", # This is crucial for the desired logfile name
                    "xerces_servername": "WINSVRXERCES01",
                    "loglevel": "INFO"
                },
                "mltune": {
                    "backend": "default_backend_from_mltune" # This will be overridden by role_hint/backend kwargs
                },
                "base": {
                    "mp_glob_base_log_path": None # Let the code derive this from EQUINRUN or default
                }
            }

    class MockMqlOverrides:
        def __init__(self):
            self.env = MockEnv()

    # Temporarily replace the actual import with the mock for testing
    import tsMqlOverrides
    tsMqlOverrides.CMqlOverrides = MockMqlOverrides

    # Ensure a dummy EQUINRUN directory structure exists for testing path resolution
    # This simulates the environment where the script expects to find logs.
    current_script_path = Path(__file__).resolve()
    equinrun_base = current_script_path.parent.parent / 'EQUINRUN'
    logdir_path = equinrun_base / 'Logdir'
    try:
        logdir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created dummy Logdir for testing: {logdir_path}", file=sys.stderr)
    except Exception as e:
        print(f"Could not create dummy Logdir for testing: {e}", file=sys.stderr)


    # Example 1: Default logging (INFO level, enabled) for TensorFlow
    print("\n--- Example 1: Default logging (INFO, enabled) for TensorFlow backend ---")
    # This should generate: .../Logdir/WINSVRXERCES01/tensorflow/tsneuropredict_app.log
    default_logger_tf = CMLogServiceSetup.initialize_logging(role_hint='tensorflow')
    default_logger_tf.debug("This DEBUG message should NOT be seen in default_client logs (INFO level).")
    default_logger_tf.info(f"This is an INFO message from default_client (TF) ({default_logger_tf.name}).")
    default_logger_tf.warning("This is a WARNING message from default_client (TF).")
    try:
        1 / 0
    except ZeroDivisionError:
        default_logger_tf.exception("An exception occurred in default_client (TF)!")
    loguru_logger.success("Loguru says: Default client (TF) operation successful!") # Loguru's global logger


    # Example 2: Debug logging enabled for PyTorch
    print("\n--- Example 2: Debug logging (DEBUG, enabled) for PyTorch backend ---")
    # This should generate: .../Logdir/WINSVRXERCES01/pytorch/tsneuropredict_app.log
    debug_logger_pt = CMLogServiceSetup.initialize_logging(role_hint='pytorch', loglevel='DEBUG')
    debug_logger_pt.debug(f"This DEBUG message SHOULD be seen in debug_client logs (PT) ({debug_logger_pt.name}).")
    debug_logger_pt.info("This is an INFO message from debug_client (PT).")
    debug_logger_pt.error("This is an ERROR message from debug_client (PT).")
    loguru_logger.info("Loguru says: Debug client (PT) operation continuing.")


    # Example 3: Logging disabled
    print("\n--- Example 3: Logging disabled ---")
    disabled_logger = CMLogServiceSetup.initialize_logging(role_hint='disabled_client', enable_logging=False)
    disabled_logger.info(f"This INFO message should NOT be seen in disabled_client logs ({disabled_logger.name}).")
    disabled_logger.debug("This DEBUG message should definitely NOT be seen in disabled_client logs.")
    # Critical messages might still go to stderr based on Loguru's disabled config
    disabled_logger.critical("This CRITICAL message might be seen if Loguru's stderr fallback is active for critical.")
    print("Check console output above for 'Logging is disabled by configuration.' message.", file=sys.stderr)

    # Example 4: Custom log file name (overriding default 'tsneuropredict_app.log')
    print("\n--- Example 4: Custom log file name (backend still 'tensorflow') ---")
    # This should generate: .../Logdir/WINSVRXERCES01/tensorflow/my_custom_client_log.log
    custom_file_logger = CMLogServiceSetup.initialize_logging(role_hint='tensorflow', logfile='my_custom_client_log.log')
    custom_file_logger.info(f"This is an INFO message for the custom log file client ({custom_file_logger.name}).")
    loguru_logger.info("Loguru says: Custom file client finished.")

    print("\n--- All examples finished. Check the 'Logdir' folder for generated log files. ---")
    print(f"Expected log directory structure under: {logdir_path}")
    print(f"  - {logdir_path}/WINSVRXERCES01/tensorflow/tsneuropredict_app.log")
    print(f"  - {logdir_path}/WINSVRXERCES01/pytorch/tsneuropredict_app.log")
    print(f"  - {logdir_path}/WINSVRXERCES01/tensorflow/my_custom_client_log.log")