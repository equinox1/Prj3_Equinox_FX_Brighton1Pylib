# filename: tsMqlLogService.py
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
from typing import Optional, Dict, Any # Added for type hinting

from tsMqlPlatform import run_platform, platform_checker
from rich.console import Console # Keep Console for potential direct use, but simplify add()
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path

# Import CMqlOverrides here, as it's used within CMLogServiceSetup
from tsMqlOverrides import CMqlOverrides

# Initialize platform checkers - these are global to the module
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()

# Rich traceback installation for better error logs
install(show_locals=True, extra_lines=10)

class CMLogServiceSetup:
    """
    Centralized logging service setup using Loguru for flexible logging.
    Manages log file paths, levels, and console output based on application parameters
    and backend selection.
    """
    # Class-level variables to store configuration and ensure single initialization
    _initialized = False
    _default_log_file_name = 'tsneuropredict_app.log'
    _default_base_log_dir = Path("./Logdir") # Fallback default

    @classmethod
    def initialize_logging(cls, app_params: Dict = None, tune_params: Dict = None,
                           base_params: Dict = None, role_hint: str = "main",
                           loglevel: str = 'INFO', enable_logging: bool = True,
                           logfile: Optional[str] = None, backend: str = "pytorch"): # Added backend parameter
        """
        Initializes the Loguru logger with dynamic configuration.
        This method is designed to be called once per process.

        :param app_params: Dictionary of application-specific parameters.
        :param tune_params: Dictionary of ML tuning parameters.
        :param base_params: Dictionary of base parameters, including global log path.
        :param role_hint: A string indicating the role of the current process (e.g., 'chief', 'worker_1', 'oracle_server').
                          Used to create distinct log file names and directories.
        :param loglevel: The minimum logging level to capture (e.g., 'INFO', 'DEBUG', 'WARNING').
        :param enable_logging: If False, logging to files will be disabled. Console logging might still occur.
        :param logfile: Optional. A specific filename for the log. If not provided,
                        it defaults to '{role_hint}_tsneuropredict_app.log'.
        :param backend: The backend type (e.g., 'tensorflow', 'pytorch'). Used for log directory structure.
        :return: The configured logger instance.
        """
        if cls._initialized:
            # logger.warning("CMLogServiceSetup already initialized. Skipping re-initialization.")
            return logging.getLogger(role_hint) # Return a logger for the specific role

        # Ensure parameters are dictionaries
        app_params = app_params if app_params is not None else {}
        tune_params = tune_params if tune_params is not None else {}
        base_params = base_params if base_params is not None else {}

        # Determine the base log directory
        # Prioritize 'mp_glob_base_log_path' from base_params
        base_log_dir = Path(base_params.get('mp_glob_base_log_path', cls._default_base_log_dir))
        
        # Determine the backend for sub-directory creation
        # The 'backend' parameter is now directly available
        # backend = tune_params.get('backend', 'pytorch') # No longer needed here as it's a parameter

        # Construct the final log directory path
        final_log_dir = base_log_dir / backend
        final_log_dir.mkdir(parents=True, exist_ok=True)

        # Determine the log file name
        # If a specific logfile is provided, use it. Otherwise, use role_hint.
        effective_logfile_name = logfile if logfile else f"{role_hint}_{app_params.get('xerces_logfile', cls._default_log_file_name)}"
        log_file_path = final_log_dir / effective_logfile_name

        # Remove all existing handlers from Loguru to start fresh
        loguru_logger.remove()

        # Add a handler for console output (stderr)
        loguru_logger.add(
            sys.stderr,
            level=loglevel.upper(),
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        # Add a handler for the log file if logging is enabled
        if enable_logging:
            loguru_logger.add(
                str(log_file_path),
                level=loglevel.upper(),
                rotation="10 MB", # Rotate file every 10 MB
                compression="zip", # Compress rotated files
                retention="7 days", # Keep logs for 7 days
                enqueue=True, # Use a queue for non-blocking logging
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
            )
            loguru_logger.info(f"Logging enabled to file: {log_file_path}")
        else:
            loguru_logger.info("Logging disabled by configuration (no file handler added).")

        # Redirect standard logging to Loguru
        logging.basicConfig(handlers=[RichHandler(console=Console(file=sys.stderr), show_time=True, show_level=True, show_path=True, enable_link_path=True)], level=loglevel.upper())
        # This line ensures that calls to the standard `logging` module
        # are routed through Loguru's configured handlers.
        logging.getLogger().handlers = [LoguruHandler()] # Ensure root logger uses LoguruHandler
        
        # Suppress loguru's default handler if it's already added
        # This is a common issue where Loguru adds a default handler to stderr
        # even if you explicitly remove and re-add.
        # It's better to manage all handlers explicitly.
        # loguru_logger.configure(handlers=[{"sink": sys.stderr, "level": loglevel.upper()}])

        # Set the flag to indicate initialization
        cls._initialized = True
        
        # Return a standard Python logger instance for the specific role,
        # which will now be managed by Loguru.
        return logging.getLogger(role_hint)

# Custom handler to bridge standard logging to Loguru
class LoguruHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelname

        frame = logging.currentframe()
        depth = 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Example usage (for testing purposes, will not run when imported as a module)
if __name__ == "__main__":
    print("--- Running CMLogServiceSetup test cases ---")

    # Example 1: Default logging
    print("\n--- Example 1: Default logging (role: main) ---")
    logger_main = CMLogServiceSetup.initialize_logging(
        app_params={'LOGLEVEL': 'INFO', 'xerces_logfile': 'my_app.log'},
        base_params={'mp_glob_base_log_path': './TestLogdir'},
        role_hint='main'
    )
    logger_main.info("This is an INFO message from the main role.")
    logger_main.debug("This DEBUG message should NOT be seen at INFO level.")
    logger_main.error("This is an ERROR message from the main role.")
    logging.getLogger("some_other_module").warning("This standard WARNING message should also be captured.")

    # Example 2: Worker logging
    print("\n--- Example 2: Worker logging (role: worker_1) ---")
    logger_worker = CMLogServiceSetup.initialize_logging(
        app_params={'LOGLEVEL': 'DEBUG', 'xerces_logfile': 'my_app.log'},
        tune_params={'backend': 'tensorflow'},
        base_params={'mp_glob_base_log_path': './TestLogdir'},
        role_hint='worker_1',
        backend='tensorflow' # Explicitly passing backend for the example
    )
    logger_worker.info("This is an INFO message from worker_1.")
    logger_worker.debug("This DEBUG message SHOULD be seen from worker_1.")

    # Example 3: Disabled logging
    print("\n--- Example 3: Disabled logging (role: disabled_client) ---")
    disabled_logger = CMLogServiceSetup.initialize_logging(role_hint='disabled_client', enable_logging=False)
    disabled_logger.info(f"This INFO message should NOT be seen in disabled_client logs ({disabled_logger.name}).")
    disabled_logger.debug("This DEBUG message should definitely NOT be seen in disabled_client logs.")
    disabled_logger.critical("This CRITICAL message might be seen if Loguru's stderr fallback is active for critical.")
    print("Check console output above for 'Logging is disabled by configuration.' message.", file=sys.stderr)

    print("\n--- All examples finished. Check the 'TestLogdir' folder for generated log files. ---")
    print(f"Expected log directory structure under: {Path('./TestLogdir')}")

