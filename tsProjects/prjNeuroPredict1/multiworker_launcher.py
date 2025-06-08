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
import logging # Import logging
from pathlib import Path # Import Path for path manipulations


from tsMqlOverrides import CMqlOverrides

# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

from tsMqlSetup import CMqlSetup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

# --- Global Configuration ---
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

# ==== PATH CONFIGURATION ====
PYTHON_EXEC = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\.venv\Scripts\python.exe"
BASE_PATH = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"

ORACLE_DAEMON_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/oracle_server_main.py")
CHIEF_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
WORKER_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

ORACLE_HOST = app_params.get('xerces_server', '192.168.1.103')
ORACLE_PORT = app_params.get('xerces_port', 9000)
ORACLE_READY_TIMEOUT = 30 # seconds

# --- Logger Setup for Master Launcher ---
# Determine the global log file and directory
global_logdir, global_logfile = setup_config.set_log_dir(
    logdir=None,
    logfile=app_params.get('xerces_logfile', 'multiworker_launcher.log'),
    servername=app_params.get('xerces_servername', "default_server"),
    backend=GLOBAL_BACKEND
)
# Configure logging for this master process

setup_config.setup_logging(logfile=global_logfile)

# Get the logger instance for this module after setup_logging has been called
logger = logging.getLogger(__name__) # Or just logging.getLogger() for the root logger

logger.info(f"🚀 Multiworker Launcher started with Global Backend: {GLOBAL_BACKEND}")
logger.info(f"All processes will log to: {global_logfile}")

# Set environment variables for child processes to inherit logging configuration
os.environ['MLTUNE_BACKEND'] = GLOBAL_BACKEND
os.environ['GLOBAL_LOGFILE_PATH'] = global_logfile
os.environ['GLOBAL_LOGDIR_PATH'] = global_logdir # Also pass the log directory

def launch_process(script_path, tuner_id, backend):
    """Launches a Python script in a new process with specific environment variables."""
    env = os.environ.copy()
    env['TUNER_ID'] = tuner_id
    env['MLTUNE_BACKEND'] = backend # Pass the backend to the child process

    # These are already in os.environ from above, but explicitly setting them again
    # ensures clarity and robustness for child processes.
    env['GLOBAL_LOGFILE_PATH'] = global_logfile
    env['GLOBAL_LOGDIR_PATH'] = global_logdir

    # Optional: Suppress TF warnings in child processes for cleaner logs
    env['TF_FORCE_UNIFIED_MEMORY'] = "1"
    env['TF_DISABLE_POOL_ALLOCATOR'] = "1"
    env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    
    # Correctly build the command list, ensuring PYTHON_EXEC is first
    command = [PYTHON_EXEC, script_path]
    
    logger.info(f"Launching {tuner_id} process: {' '.join(command)}")
    logger.debug(f"  Environment passed: TUNER_ID={env['TUNER_ID']}, MLTUNE_BACKEND={env['MLTUNE_BACKEND']}")
    logger.debug(f"  Child process logging to: {env['GLOBAL_LOGFILE_PATH']}")
    
    # Configure startupinfo for Windows to hide the console window
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE # Hide the window

    # Use Popen to allow non-blocking execution, with hidden console on Windows
    return subprocess.Popen(command, env=env, startupinfo=startupinfo)

def port_in_use(host, port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            logger.warning(f"Port {port} is already in use by another process.")
            return True

def kill_process_on_port(port):
    """Find and kill the process using the given port."""
    logger.info(f"Attempting to kill processes on port {port}...")
    killed_any = False
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.pid is not None:
            try:
                p = psutil.Process(conn.pid)
                p.terminate() # or p.kill() for more forceful termination
                p.wait(timeout=5) # Wait for process to terminate
                logger.info(f"Killed process {conn.pid} (Name: {p.name()}) on port {port}")
                killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                logger.warning(f"Could not kill process {conn.pid} on port {port} (already terminated or access denied).")
            except Exception as e:
                logger.error(f"Error killing process {conn.pid} on port {port}: {e}")
    if not killed_any:
        logger.info(f"No active processes found or killed on port {port}.")
    return killed_any

def wait_for_oracle_ready(timeout=ORACLE_READY_TIMEOUT):
    """Waits for the OracleServer to become responsive."""
    start_time = time.time()
    logger.info(f"Waiting for OracleServer to become ready at http://{ORACLE_HOST}:{ORACLE_PORT} (timeout: {timeout}s)...")
    while time.time() - start_time < timeout:
        try:
            # Check if OracleServer is responsive
            response = requests.get(f"http://{ORACLE_HOST}:{ORACLE_PORT}/list_trials", timeout=5)
            if response.status_code == 200:
                logger.info("✅ OracleServer is ready and responsive.")
                return True
        except requests.ConnectionError:
            logger.debug("OracleServer not yet reachable. Retrying...")
        except Exception as e:
            logger.error(f"Error checking OracleServer status: {e}", exc_info=True)
        time.sleep(2) # Wait before retrying
    logger.error(f"❌ OracleServer not responsive within {timeout} seconds. Aborting startup.")
    return False

# ==== MAIN ENTRYPOINT ====
if __name__ == "__main__":
    oracle_proc = None
    try:
        if port_in_use(ORACLE_HOST, ORACLE_PORT):
            if FORCE_KILL:
                logger.warning(f"Port {ORACLE_PORT} in use. Attempting forced shutdown and restart...")
                kill_process_on_port(ORACLE_PORT)
                time.sleep(2) # Give a moment for port to free up
                if port_in_use(ORACLE_HOST, ORACLE_PORT): # Re-check after kill attempt
                    logger.error(f"Port {ORACLE_PORT} still in use after kill attempt. Cannot proceed.")
                    sys.exit(1)
            else:
                logger.info(f"Port {ORACLE_PORT} already in use. Assuming Oracle is running. Not launching a new one.")
                # We don't launch oracle_proc, so set to None
        
        # Only launch Oracle if it's not assumed to be running or was successfully killed
        if not port_in_use(ORACLE_HOST, ORACLE_PORT):
            logger.info("🚀 Launching OracleServer Daemon...")
            oracle_proc = launch_process(ORACLE_DAEMON_SCRIPT, tuner_id="oracle", backend=GLOBAL_BACKEND)
            if not wait_for_oracle_ready():
                logger.critical("❌ Aborting: OracleServer failed to start or become ready.")
                if oracle_proc:
                    oracle_proc.terminate()
                    oracle_proc.wait()
                sys.exit(1)
        
        # If oracle_proc is None, it means Oracle was already running or not launched.
        # Ensure that wait_for_oracle_ready is called if no new process was started
        # to confirm responsiveness of the existing Oracle.
        elif oracle_proc is None and not wait_for_oracle_ready():
             logger.critical("❌ Existing OracleServer not responsive. Aborting.")
             sys.exit(1)

        logger.info("👑 Launching Chief Process...")
        chief_proc = launch_process(CHIEF_SCRIPT, tuner_id="chief", backend=GLOBAL_BACKEND)

        logger.info("🧑‍🔬 Launching Worker Processes...")
        workers = [launch_process(WORKER_SCRIPT, tuner_id=f"worker{i+1}", backend=GLOBAL_BACKEND) for i in range(NUM_WORKERS)]

        logger.info("Main processes launched. Waiting for chief process to complete...")
        chief_proc.wait()
        logger.info("Chief process completed. Shutting down worker processes.")

    except KeyboardInterrupt:
        logger.info("🚫 KeyboardInterrupt received in launcher. Initiating graceful shutdown...")
    except Exception as e:
        logger.exception(f"An unhandled error occurred in multiworker_launcher: {e}")
    finally:
        # Ensure all child processes are terminated
        if 'chief_proc' in locals() and chief_proc and chief_proc.poll() is None:
            logger.info("Terminating Chief process...")
            chief_proc.terminate()
            chief_proc.wait(timeout=10)
        if 'workers' in locals():
            for i, worker in enumerate(workers):
                if worker and worker.poll() is None:
                    logger.info(f"Terminating Worker {i+1} process...")
                    worker.terminate()
                    worker.wait(timeout=10)
        if oracle_proc and oracle_proc.poll() is None:
            logger.info("Terminating OracleServer process...")
            oracle_proc.terminate()
            oracle_proc.wait(timeout=10)
        logger.info("All child processes ensured terminated. Launcher shutting down.")

