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

# Set up logging for the launcher itself
# Need to initialize CMqlSetup again for the launcher process's logger
launcher_log_config = CMqlSetup()
launcher_log_config.setup_logging(logfile=str(global_logfile))
logger = logging.getLogger(__name__) # Get logger for the launcher script

logger.info(f"🚀 Multiworker Launcher started with Global Backend: {GLOBAL_BACKEND}")
logger.info(f"All processes will log to: {global_logfile}")
logger.info(f"Log directory set to: {global_logdir}")
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
ORACLE_READY_TIMEOUT = 60 # Seconds to wait for Oracle Server to become ready

PYTHON_EXEC = sys.executable # Path to the current Python interpreter

def is_port_in_use(port):
    """Checks if a given port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('xerces_server', port)) == 0

def wait_for_oracle_ready():
    """Waits for the Oracle Server to become responsive."""
    logger.info(f"Waiting for OracleServer to be ready at "
                f"http://{ORACLE_HOST}:{ORACLE_PORT}...")
    start_time = time.time()
    while time.time() - start_time < ORACLE_READY_TIMEOUT:
        try:
            response = requests.get(f"http://{ORACLE_HOST}:{ORACLE_PORT}/list_trials",
                                    timeout=5)
            if response.status_code == 200:
                logger.info("✅ OracleServer is ready.")
                return True
        except requests.exceptions.ConnectionError:
            logger.debug(f"OracleServer not yet ready, retrying in 2 seconds...")
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
                logger.info(f"Terminating Worker process {worker.pid}...")
                terminate_process_and_children(worker.pid)
        if oracle_proc and oracle_proc.poll() is None:
            logger.info("Terminating OracleServer process...")
            terminate_process_and_children(oracle_proc.pid)
        
        logger.info("All child processes ensured terminated. Launcher shutting down.")
