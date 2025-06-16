runtune="pt" # tf or pt
# Determine the global backend based on runtune
GLOBAL_BACKEND = "tensorflow" if runtune == "tf" else "pytorch"
FORCE_KILL = True
NUM_WORKERS = 1  # Number of worker processes to launch
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
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=_estimated_physical_cores
)


# Import environment manager and overrides after logging setup
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides

mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__,
    loglevel='INFO',
    # Explicitly set the logfile name to ensure consistency
    logfile='tsneuropredict_app.log',
    # Pass the determined backend so logging goes into the correct subdirectory
    backend=GLOBAL_BACKEND # Pass the backend to the logging setup
)



# Use the configured values for server and port
ORACLE_SERVER_HOST = app_params.get('xerces_server', '192.168.1.103')
ORACLE_SERVER_PORT = app_params.get('xerces_port', 9000)

CHIEF_SCRIPT = "tsNeuroPredictWinMql_chief.py"
WORKER_SCRIPT = "tsNeuroPredictWinMql_worker.py"
ORACLE_SERVER_SCRIPT = "oracle_server_main.py"

# Function to check if a port is in use
def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True

# Function to terminate a process and its children
def terminate_process_and_children(pid):
    try:
        process = psutil.Process(pid)
        for child in process.children(recursive=True):
            child.terminate()
        process.terminate()
        logger.info(f"Terminated process {pid} and its children.")
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} already terminated or does not exist.")
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}", exc_info=True)


def launch_process(script_name, tuner_id=None, backend=None):
    """Launches a Python script as a subprocess."""
    env = os.environ.copy()
    if tuner_id:
        env['TUNER_ID'] = tuner_id
        logger.info(f"Setting TUNER_ID={tuner_id} for {script_name}")
    if backend:
        env['BACKEND'] = backend
        logger.info(f"Setting BACKEND={backend} for {script_name}")

    # Pass the Oracle server details as environment variables
    env['ORACLE_SERVER_HOST'] = ORACLE_SERVER_HOST
    env['ORACLE_SERVER_PORT'] = str(ORACLE_SERVER_PORT)

    # REMOVED: No longer explicitly pass LOGDIR and LOGFILE.
    # Child processes will call CMLogServiceSetup.initialize_logging and
    # derive these paths based on backend and their own logfile argument.
    # env['LOGDIR'] = str(setup_config.global_logdir)
    # env['LOGFILE'] = setup_config.global_logfile

    # IMPORTANT: Ensure all environment variables are strings
    # Iterate over a copy of items to allow modification during iteration
    for key, value in list(env.items()):
        env[key] = str(value)

    python_executable = sys.executable # Use the current Python interpreter

    # Use Path objects for script paths for better OS compatibility
    script_path = Path(__file__).parent / script_name

    command = [python_executable, str(script_path)]
    logger.info(f"Launching process: {command} with env TUNER_ID={env.get('TUNER_ID')}, BACKEND={env.get('BACKEND')}")

    # Redirect stdout and stderr to a pipe for unified logging
    process = subprocess.Popen(
        command,
        env=env, # Pass the sanitized environment
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, # Decode stdout/stderr as text
        bufsize=1, # Line-buffered
        universal_newlines=True # Handle different line endings
    )

    # Start a thread to read and log the output
    def log_stream(stream, process_name):
        for line in iter(stream.readline, ''):
            # Add prefix to distinguish logs from different processes
            # Remove extra newlines that might be added by print() in subprocesses
            logged_line = f"[{process_name}] {line.strip()}"
            if "INFO:" in logged_line:
                logger.info(logged_line)
            elif "WARNING:" in logged_line:
                logger.warning(logged_line)
            elif "ERROR:" in logged_line:
                logger.error(logged_line)
            elif "CRITICAL:" in logged_line:
                logger.critical(logged_line)
            elif "DEBUG:" in logged_line:
                logger.debug(logged_line)
            else:
                logger.info(logged_line) # Default to info for unclassified lines
        stream.close()

    # Start separate threads for stdout and stderr to prevent blocking
    threading.Thread(target=log_stream, args=(process.stdout, script_name.split('.')[0]), daemon=True).start()
    # If stderr is also piped to STDOUT, no need for a separate stderr thread
    # threading.Thread(target=log_stream, args=(process.stderr, f"{script_name.split('.')[0]}_ERR"), daemon=True).start()

    return process

def wait_for_server(host, port, timeout=60, interval=2):
    """
    Waits for the server to become available at the given host and port.
    Performs a health check with retries.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/status", timeout=5)
            response.raise_for_status()
            logger.info(f"Oracle Server health check successful: {response.json()}")
            return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection refused to Oracle Server at {host}:{port}. Retrying in {interval} seconds...")
            time.sleep(interval)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during Oracle Server health check: {e}", exc_info=True)
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Unexpected error during server health check: {e}", exc_info=True)
            time.sleep(interval)
    logger.error(f"Oracle Server did not become available at {host}:{port} within {timeout} seconds.")
    return False


if __name__ == "__main__":
    chief_proc = None
    workers = []
    oracle_proc = None

    try:
        # 1. Launch Oracle Server (if not already running)
        logger.info(f"Checking if Oracle Server is running on {ORACLE_SERVER_HOST}:{ORACLE_SERVER_PORT}...")
        if is_port_in_use(ORACLE_SERVER_HOST, ORACLE_SERVER_PORT):
            logger.info(f"Port {ORACLE_SERVER_PORT} is already in use. Assuming Oracle Server is running.")
            # If the server is already running, we don't need to launch it.
            # We also don't have its PID, so we won't try to terminate it later.
        else:
            logger.info(f"Port {ORACLE_SERVER_PORT} is free. Launching Oracle Server...")
            oracle_proc = launch_process(ORACLE_SERVER_SCRIPT, tuner_id="oracle_main_server", backend=GLOBAL_BACKEND)

            # Wait for the Oracle Server to start up using the robust retry mechanism
            if not wait_for_server(ORACLE_SERVER_HOST, ORACLE_SERVER_PORT, timeout=60, interval=5): # Increased interval
                logger.critical("Oracle Server failed to start or respond within the allocated time. Exiting.")
                if oracle_proc and oracle_proc.poll() is None:
                    terminate_process_and_children(oracle_proc.pid)
                sys.exit(1)


        # 2. Launch Chief process
        logger.info(f"Launching Chief process: {CHIEF_SCRIPT}")
        chief_proc = launch_process(CHIEF_SCRIPT, tuner_id="chief", backend=GLOBAL_BACKEND)

        # 3. Launch Worker processes
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
        if oracle_proc and oracle_proc.poll() is None:
            logger.info("Terminating OracleServer process...\n") # Added newline
            terminate_process_and_children(oracle_proc.pid)

        logger.info("All child processes ensured terminated. Launcher shutting down.")