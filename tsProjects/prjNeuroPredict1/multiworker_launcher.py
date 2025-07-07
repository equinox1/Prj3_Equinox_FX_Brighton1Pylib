runtune="pt" # tf or pt
# Determine the global backend based on runtune
GLOBAL_BACKEND = "tensorflow" if runtune == "tf" else "pytorch"
FORCE_KILL = True
NUM_WORKERS = 2  # Increased to 2 for better demonstration of multi-worker tuning
import subprocess
import time
import os
import socket
import requests
import sys
import psutil
import logging
from pathlib import Path
import threading


import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from tsMqlSetup import CMqlSetup
# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.)
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

from tsMqlLogService import CMLogServiceSetup # Import the CMLogServiceSetup class

# Determine backend for the launcher's own logging
# This should directly use GLOBAL_BACKEND, not rely on os.environ.get which might be unset
launcher_backend_for_log = GLOBAL_BACKEND

# Initialize master logging for the launcher process
# All logs from all processes should eventually go to this central file
logger = CMLogServiceSetup.initialize_logging(
    role_hint='launcher',
    loglevel='INFO', # Set desired log level for the launcher
    backend=launcher_backend_for_log, # Pass the determined GLOBAL_BACKEND
    # logfile='tsneuro_predict.log', # This is now handled internally by CMLogServiceSetup
    # logdir will be picked up from base_params or derived
)

# Define scripts paths relative to the launcher script's directory
LAUNCHER_DIR = Path(__file__).parent
ORACLE_SCRIPT = LAUNCHER_DIR / "oracle_server_main.py"
CHIEF_SCRIPT = LAUNCHER_DIR / "tsNeuroPredictWinMql_chief.py"
WORKER_SCRIPT = LAUNCHER_DIR / "tsNeuroPredictWinMql_worker.py"

# Ensure the scripts exist
for script in [ORACLE_SCRIPT, CHIEF_SCRIPT, WORKER_SCRIPT]:
    if not script.exists():
        logger.error(f"Required script not found: {script}")
        sys.exit(1)

def is_port_in_use(port, host='127.0.0.1', retries=5, delay=1):
    """
    Check if a port is already in use with retries.
    Args:
        port (int): The port number to check.
        host (str): The host address to check.
        retries (int): Number of times to retry checking the port.
        delay (int): Delay in seconds between retries.
    Returns:
        bool: True if the port is in use, False otherwise.
    """
    for i in range(retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                # If bind is successful, port is free, so close and return False
                s.close()
                return False
            except socket.error as e:
                # If error is EADDRINUSE, port is in use
                if e.errno == errno.EADDRINUSE:
                    logger.warning(f"Port {host}:{port} is in use (Attempt {i+1}/{retries}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Other socket errors
                    logger.error(f"Socket error checking port {host}:{port}: {e}", exc_info=True)
                    return True # Treat other errors as port being problematic
    logger.error(f"Port {host}:{port} remained in use after {retries} attempts.")
    return True # Port is still in use after all retries


def terminate_process_and_children(pid):
    """Terminates a process and its entire child process tree."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            if child.is_running():
                logger.info(f"Terminating child process {child.pid}...")
                child.terminate()
        if parent.is_running():
            logger.info(f"Terminating parent process {parent.pid}...")
            parent.terminate()
        gone, alive = psutil.wait_procs(children + [parent], timeout=5)
        for p in alive:
            if p.is_running():
                logger.info(f"Killing stubborn process {p.pid}...")
                p.kill()
        logger.info(f"Terminated process {pid} and its children.")
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} not found (already terminated).")
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}", exc_info=True)

def launch_process(script_path, tuner_id=None, backend=None, is_oracle=False, is_chief=False):
    """Launches a Python script in a new process."""
    cmd = [sys.executable, str(script_path)]
    env = os.environ.copy()

    # Set environment variables for the subprocess
    env['BACKEND'] = backend if backend else GLOBAL_BACKEND
    env['TF_CPP_MIN_LOG_LEVEL'] = '1' # Suppress TensorFlow info/warning logs in subprocesses

    # Pass tuning parameters via environment variables for consistency
    # These values will override any defaults in the config files
    env['MLTUNE_NUM_TRIALS'] = '100' # Increased number of trials
    env['MLTUNE_MAX_EPOCHS'] = '20' # Increased max epochs per trial

    if tuner_id:
        env['TUNER_ID'] = tuner_id
    if is_chief:
        env['IS_CHIEF'] = 'True' # Indicate if it's the chief process
    if is_oracle:
        # Crucial: Force overwrite for the Oracle Server to ensure a fresh start
        env['MLTUNE_OVERWRITE'] = 'True'
        logger.info("Setting MLTUNE_OVERWRITE=True for Oracle Server process.")


    # Ensure the correct Python environment is used if running in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env['PATH']
        env['VIRTUAL_ENV'] = sys.prefix

    logger.info(f"Launching command: {cmd} with env BACKEND={env['BACKEND']}, TUNER_ID={env.get('TUNER_ID')}, IS_CHIEF={env.get('IS_CHIEF')}, MLTUNE_NUM_TRIALS={env.get('MLTUNE_NUM_TRIALS')}, MLTUNE_MAX_EPOCHS={env.get('MLTUNE_MAX_EPOCHS')}, MLTUNE_OVERWRITE={env.get('MLTUNE_OVERWRITE')}")
    
    # Use Popen for non-blocking launch
    process = subprocess.Popen(cmd, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return process

if __name__ == "__main__":
    oracle_proc = None
    chief_proc = None
    workers = []

    try:
        # Check if OracleServer port is in use
        oracle_port = 9000 # Default port, should match app_params in config
        oracle_host = '127.0.0.1' # Use localhost for binding check

        if is_port_in_use(oracle_port, host=oracle_host):
            logger.warning(f"Port {oracle_host}:{oracle_port} is already in use. Assuming OracleServer is already running and skipping launch.")
            # If port is in use, we don't launch a new OracleServer
            oracle_proc = None
        else:
            # Launch OracleServer
            oracle_proc = launch_process(ORACLE_SCRIPT, is_oracle=True)
            logger.info(f"🚀 Launched OracleServer with PID: {oracle_proc.pid}")
            time.sleep(5) # Give OracleServer time to start

        # Launch Chief process
        chief_proc = launch_process(CHIEF_SCRIPT, tuner_id="chief",
                                    backend=GLOBAL_BACKEND, is_chief=True)
        logger.info(f"🚀 Launched Chief process with PID: {chief_proc.pid}")
        time.sleep(5) # Give Chief time to start and register with Oracle

        # Launch Worker processes
        logger.info(f"Launching {NUM_WORKERS} worker processes for backend: {GLOBAL_BACKEND}")
        workers = [launch_process(WORKER_SCRIPT, tuner_id=f"worker{i+1}",
                                  backend=GLOBAL_BACKEND) for i in range(NUM_WORKERS)]

        # Wait for the chief process to complete
        # This will block until chief_proc exits
        if chief_proc:
            chief_proc.wait()
            logger.info("Chief process completed. Shutting down worker processes.")
        else:
            logger.info("Chief process was not launched by this script or already completed.")

    except KeyboardInterrupt:
        logger.info("🚫 KeyboardInterrupt received. Shutting down processes...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Terminate all launched processes
        if chief_proc and chief_proc.poll() is None:
            logger.info("Terminating Chief process...")
            terminate_process_and_children(chief_proc.pid)
        for worker in workers:
            if worker and worker.poll() is None:
                logger.info(f"Terminating Worker process {worker.pid}...\n") # Added newline
                terminate_process_and_children(worker.pid)
        # Only attempt to terminate oracle_proc if it was actually launched by this script
        if oracle_proc and oracle_proc.poll() is None:
            logger.info("Terminating OracleServer process...\n") # Added newline
            terminate_process_and_children(oracle_proc.pid)

        logger.info("All child processes ensured terminated. Launcher shutting down.")
