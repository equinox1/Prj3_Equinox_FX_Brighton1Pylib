<<<<<<< HEAD
runtune="pt" # tf or pt
# Determine the global backend based on runtune
GLOBAL_BACKEND = "tensorflow" if runtune == "tf" else "pytorch"
FORCE_KILL = True
NUM_WORKERS = 1  # Number of worker processes to launch
=======
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
import subprocess
import time
import os
import socket
import requests
import sys
import psutil
import logging
<<<<<<< HEAD
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
    num_threads=1
)
# --- Global Configuration ---
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
# Get all parameters after overrides
env_mgr = mql_overrides.env
all_params = env_mgr.all_params()

app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
base_params = all_params.get('base', {})

xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# Determine global logfile and logdir based on config
global_log_base_path = all_params.get('base', {}).get('mp_glob_base_log_path', 'Logdir')
global_logfile = Path(global_log_base_path) / xerces_servername / \
                 GLOBAL_BACKEND / xerces_logfile
global_logdir = Path(global_log_base_path) / xerces_servername / GLOBAL_BACKEND

# Ensure log directories exist
global_logdir.mkdir(parents=True, exist_ok=True)
# Ensure parent directory for logfile exists (Path.parent returns the parent directory)
global_logfile.parent.mkdir(parents=True, exist_ok=True) 

# Set environment variables for child processes to inherit logging configuration
os.environ['GLOBAL_LOGFILE_PATH'] = str(global_logfile)
os.environ['GLOBAL_LOGDIR_PATH'] = str(global_logdir)
os.environ['MLTUNE_BACKEND'] = GLOBAL_BACKEND
os.environ['MLTUNE_TUNER_TYPE'] = tune_params.get('tuner_type', 'hyperband')
os.environ['MLTUNE_TRIALS'] = str(tune_params.get('num_trials', 128))

# Initialize CMqlSetup for logging.
# This block ensures that the root logger is configured only once across the application,
# preventing duplicate log messages or repeated file handler creations.
root_logger = logging.getLogger()
logger_initial_print_message = "" # Initialize message
if not root_logger.handlers: # Check if the root logger has any handlers configured already
    # If no handlers exist, proceed with setting up the logging
    clientlog_config = CMqlSetup()
    try:
        # Determine the final log directory and file path
        final_logdir, final_logfile_path = clientlog_config.set_log_dir(
            logdir=app_params.get('LOGDIR'), # Use LOGDIR from app_params if available
            logfile=app_params.get('LOGFILE', 'tsneuropredict_app.log'), # Use LOGFILE from app_params or default
            servername=xerces_servername,
            backend=backend
        )
        # Configure the logging system
        clientlog_config.setup_logging(logfile=final_logfile_path)
        logger_initial_print_message = f"Logging initialized. Logfile: {final_logfile_path}"
    except Exception as e:
        # If logging setup fails, print a critical error to stderr and exit
        print(f"CRITICAL ERROR: Failed to set up log directories or configure logging via CMqlSetup: {e}", file=sys.stderr)
        sys.exit(1)
else:
    # If handlers already exist, it means logging was configured by another module.
    # In this case, we skip re-initialization to avoid issues.
    logger_initial_print_message = "Logging already configured. Skipping re-initialization."

# Get the logger instance for this specific module.
# This should always be done AFTER the root logger has been configured.
logger = logging.getLogger(__name__)
# Log the initial message, confirming whether logging was set up or already existed.
logger.info(logger_initial_print_message)

# Define servername before using it in the logging statement
servername = xerces_servername # Assign the value from app_params
logger.debug(f"Servername: {servername}")

# Define scripts relative to the current file
# Assuming this script is in PythonLib/tsProjects/prjNeuroPredict1
BASE_PATH = Path(__file__).resolve().parent.parent.parent
CHIEF_SCRIPT = BASE_PATH / 'tsProjects' / 'prjNeuroPredict1' / \
               'tsNeuroPredictWinMql_chief.py'
WORKER_SCRIPT = BASE_PATH / 'tsProjects' / 'prjNeuroPredict1' / \
                'tsNeuroPredictWinMql_worker.py'
# Path to your OracleServer daemon script
ORACLE_DAEMON_SCRIPT = BASE_PATH / 'tsProjects' / 'prjNeuroPredict1' / \
                       'oracle_server_main.py'

ORACLE_HOST = xerces_server # Use the IP from config
ORACLE_PORT = xerces_port # Use the port from config
ORACLE_READY_TIMEOUT = 30 # Seconds to wait for Oracle Server to become ready

PYTHON_EXEC = sys.executable # Path to the current Python interpreter

def is_port_in_use(port):
    """Checks if a given port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((xerces_server, port)) == 0

def wait_for_oracle_ready():
    """Waits for the Oracle Server to become responsive using the /status endpoint."""
    logger.info(f"Waiting for OracleServer to be ready at "
                f"http://{ORACLE_HOST}:{ORACLE_PORT}/status...")
    start_time = time.time()
    while time.time() - start_time < ORACLE_READY_TIMEOUT:
        try:
            # CORRECTED: Use /status endpoint for health check
            response = requests.get(f"http://{ORACLE_HOST}:{ORACLE_PORT}/status",
                                    timeout=5)
            if response.status_code == 200 and response.json().get("status") == "Oracle Server is running.":
                logger.info("✅ OracleServer is ready.")
                return True
            else:
                logger.debug(f"OracleServer /status check returned: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"OracleServer not yet ready (connection error), retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error checking OracleServer status: {e}")
            time.sleep(2)
    logger.error("❌ OracleServer did not become ready within the timeout period.")
    return False

def terminate_process_and_children(pid):
    """Terminates a process and its children recursively."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            if FORCE_KILL:
                child.kill()
            else:
                child.terminate()
        if FORCE_KILL:
            parent.kill()
        else:
            parent.terminate()
        gone, alive = psutil.wait_procs(children + [parent], timeout=5)
        for p in alive:
            logger.warning(f"Process {p.pid} ({p.name()}) did not terminate, forcing kill.")
            p.kill()
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} already terminated.")
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}")

def launch_process(script_path, tuner_id, backend):
    """Launches a Python script as a subprocess."""
    logger.info(f"Launching {script_path.name} with TUNER_ID={tuner_id}, "
                f"MLTUNE_BACKEND={backend}")
    env = os.environ.copy()
    env['TUNER_ID'] = tuner_id
    env['MLTUNE_BACKEND'] = backend
    # Ensure child processes use the same logfile and logdir
    env['GLOBAL_LOGFILE_PATH'] = str(global_logfile)
    env['GLOBAL_LOGDIR_PATH'] = str(global_logdir)

    # Explicitly pass the full path to the Python executable
    process = subprocess.Popen([PYTHON_EXEC, str(script_path)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Start a thread to read stdout/stderr to prevent deadlock and log output
    def log_output(process_obj, process_name):
        for line in process_obj.stdout:
            # Filter out known TensorFlow/PyTorch verbose output if not in debug mode
            if "I tensorflow/compiler" in line or "E tensorflow/compiler" in line:
                logger.debug(f"[{process_name} TF/PT Output] {line.strip()}")
            else:
                logger.info(f"[{process_name}] {line.strip()}")
        process_obj.stdout.close()

    threading.Thread(target=log_output, args=(process, script_path.name),
                     daemon=True).start()
    
    logger.info(f"Started {script_path.name} with PID: {process.pid}")
    return process

if __name__ == "__main__":
    oracle_proc = None
    chief_proc = None
    workers = []

    try:
        # Check if Oracle Server is already running
        if is_port_in_use(ORACLE_PORT):
            logger.info(f"⚠️ Port {ORACLE_PORT} already in use. Assuming Oracle is running.")
            # We don't need to launch it, just ensure it's responsive
            if not wait_for_oracle_ready():
                logger.error("❌ Aborting: Existing OracleServer at port "
                             f"{ORACLE_PORT} is not responsive.")
                sys.exit(1)
            oracle_proc = None # No process to manage if it's already running
        else:
            logger.info("🚀 Launching OracleServer Daemon...")
            oracle_proc = launch_process(ORACLE_DAEMON_SCRIPT, tuner_id="oracle",
                                         backend=GLOBAL_BACKEND)
            if not wait_for_oracle_ready():
                logger.error("❌ Aborting: OracleServer failed to start.")
                if oracle_proc:
                    terminate_process_and_children(oracle_proc.pid)
                sys.exit(1)
        
        logger.info("👑 Launching Chief Process...")
        logger.info(f"Using GLOBAL_BACKEND: {GLOBAL_BACKEND}")
        chief_proc = launch_process(CHIEF_SCRIPT, tuner_id="chief",
                                    backend=GLOBAL_BACKEND)

        logger.info("🧑‍🔬 Launching Worker Processes...")
        logger.info(f"Using GLOBAL_BACKEND: {GLOBAL_BACKEND}")
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

=======

from tsMqlSetup import CMqlSetup

# ==== CONFIGURATION ====
NUM_WORKERS = 8
PYTHON_EXEC = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\.venv\Scripts\python.exe"
BASE_PATH = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"

ORACLE_DAEMON_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/oracle_server_main.py")
CHIEF_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
WORKER_SCRIPT = os.path.join(BASE_PATH, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

ORACLE_HOST = '192.168.1.103'
ORACLE_PORT = 9000
ORACLE_URL = f"http://{ORACLE_HOST}:{ORACLE_PORT}"

MAX_WAIT_SECONDS = 15
MAX_RETRIES = 15
FORCE_KILL = '--force' in sys.argv

GLOBAL_BACKEND = "tensorflow"  # Options: 'pytorch', 'tensorflow'

# ==== UTILITIES ====
def port_in_use(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (OSError, socket.timeout):
        return False

def kill_process_on_port(port):
    print(f"🔍 Scanning for processes using port {port}...")
    for proc in psutil.process_iter(['pid', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    print(f"🔪 Killing process {proc.pid} using port {port}...")
                    proc.kill()
                    time.sleep(1)
        except Exception:
            continue

def launch_process(script_path, tuner_id=None, backend="tensorflow"):
    env = os.environ.copy()
    if tuner_id:
        env["TUNER_ID"] = tuner_id
    env["GTUNER_MODEL"] = backend
    env["MLTUNE_BACKEND"] = backend
    label = tuner_id.upper() if tuner_id else "PROCESS"
    print(f"[LAUNCH] Launching {label} → {script_path}")
    return subprocess.Popen([PYTHON_EXEC, script_path], env=env)

def wait_for_oracle_ready():
    print(f"⏳ Waiting for OracleServer at {ORACLE_URL}...")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(f"{ORACLE_URL}/heartbeat", timeout=MAX_WAIT_SECONDS)
            if resp.status_code == 200:
                print("✅ OracleServer heartbeat OK.")
                try:
                    gt_resp = requests.get(f"{ORACLE_URL}/get_trial", timeout=MAX_WAIT_SECONDS)
                    if gt_resp.status_code == 200:
                        print("✅ OracleServer /get_trial responsive.")
                        return True
                except Exception as e:
                    print(f"[WAIT] Oracle heartbeat OK but /get_trial failed: {e}")
            else:
                print(f"❌ Unexpected status: {resp.status_code}")
        except Exception as e:
            print(f"[WAIT] Oracle not ready yet: {e}")
        time.sleep(MAX_WAIT_SECONDS)

    print("❌ ERROR: OracleServer not responsive within timeout.")
    return False

# ==== MAIN ENTRYPOINT ====
if __name__ == "__main__":
    if port_in_use(ORACLE_HOST, ORACLE_PORT):
        if FORCE_KILL:
            print(f"⚠️ Port {ORACLE_PORT} in use. Attempting forced shutdown...")
            kill_process_on_port(ORACLE_PORT)
            time.sleep(2)
        else:
            print(f"⚠️ Port {ORACLE_PORT} already in use. Assuming Oracle is running.")
            oracle_proc = None
    else:
        print("🚀 Launching OracleServer Daemon...")
        oracle_proc = launch_process(ORACLE_DAEMON_SCRIPT, tuner_id="oracle", backend=GLOBAL_BACKEND)
        if not wait_for_oracle_ready():
            print("❌ Aborting: OracleServer failed to start.")
            if oracle_proc:
                oracle_proc.terminate()
                oracle_proc.wait()
            sys.exit(1)

    print("👑 Launching Chief Process...")
    chief_proc = launch_process(CHIEF_SCRIPT, tuner_id="chief", backend=GLOBAL_BACKEND)

    print("🧑‍🏭 Launching Worker Processes...")
    workers = []
    for i in range(NUM_WORKERS):
        tuner_id = f"worker_{i+1}"
        proc = launch_process(WORKER_SCRIPT, tuner_id=tuner_id, backend=GLOBAL_BACKEND)
        workers.append(proc)

    try:
        if chief_proc:
            chief_proc.wait()
        else:
            while True:
                time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Interrupt received. Terminating all processes...")
        if oracle_proc:
            oracle_proc.terminate()
        if chief_proc:
            chief_proc.terminate()
        for w in workers:
            w.terminate()
        print("✅ All subprocesses terminated cleanly.")
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
