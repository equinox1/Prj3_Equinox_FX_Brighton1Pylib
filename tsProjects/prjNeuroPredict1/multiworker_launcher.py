# filename: multiworker_launcher.py
# Rewritten: patched_oracle_server_main.py
# --- Backend Selection Variable ---
# Define the default backend here. Change this variable to switch between backends.
DEFAULT_BACKEND = 'tensorflow' # Options: 'tensorflow', 'pytorch'

import os
import sys
import time
from pathlib import Path
import logging
import warnings
import inspect
import socket
from fastapi import FastAPI

import uvicorn
from tsMqlOverrides import CMqlOverrides

import multiprocessing
import subprocess
import requests
# import argparse # Removed argparse

# --- PATH CONFIGURATION FOR MODULE IMPORTS ---\
# Assuming multiworker_launcher.py is in C:\...\EQUINRUN\PythonLib\tsProjects
# And common modules like tsMqlBrokerConfig are in C:\...\EQUINRUN\PythonLib
# So, we need to add the parent directory of this script (PythonLib) to sys.path.
# Path(__file__).parent gives C:\...\EQUINRUN\PythonLib\tsProjects
# Path(__file__).parent.parent gives C:\...\EQUINRUN\PythonLib
sys.path.append(str(Path(__file__).parent.parent))
# Also add the current directory (tsProjects) if it contains modules that are imported directly
sys.path.append(str(Path(__file__).parent))

# Import the chief and worker task functions
# These imports will now succeed because PythonLib is on sys.path
from tsNeuroPredictWinMql_chief import run_chief_process_task
from tsNeuroPredictWinMql_worker import run_worker_process_task

# Setup logger for the launcher itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load configuration for the launcher (and for passing to child processes)
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})


# Override backend in tune_params with the defined variable
tune_params['backend'] = DEFAULT_BACKEND
logger.info(f"Selected backend: {DEFAULT_BACKEND}")


# --- Oracle Server Configuration ---
# Use the IP from app_params, default to localhost if not found or empty
oracle_server_ip = app_params.get('xerces_server', '127.0.0.1')
oracle_port = app_params.get('xerces_port', 9000)
oracle_url = f"http://{oracle_server_ip}:{oracle_port}"

logger.info(f"Using Xerces server: {oracle_server_ip}, port: {oracle_port}")

# Set the ORACLE_URL environment variable for child processes
os.environ["ORACLE_URL"] = oracle_url
# Set the BACKEND environment variable for child processes
os.environ["BACKEND"] = DEFAULT_BACKEND # Use the defined variable

# Define paths for chief and worker scripts
chief_script_path = Path(__file__).parent / "tsNeuroPredictWinMql_chief.py"
worker_script_path = Path(__file__).parent / "tsNeuroPredictWinMql_worker.py"

# Define the log directory for the Oracle server
# This should align with how tsMqlLogService determines the log path
log_dir = Path(base_params.get('mp_glob_base_log_path'))
oracle_log_file = log_dir / tune_params.get('backend', 'pytorch') / "oracle_server.log"
oracle_log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure backend-specific log directory exists

# Define the Oracle data directory, ensuring it's unique per backend/model_id
oracle_data_dir = log_dir / "oracle_server_data" / f"{app_params.get('mp_app_model_id', 'default_model')}_{tune_params.get('backend', 'pytorch')}"
oracle_data_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Oracle data directory: {oracle_data_dir}")

# --- Start Oracle Server ---
# The Oracle server needs to be started as a separate process because it's a FastAPI app.
# We pass the oracle_data_dir to the server via an environment variable or command line.
# For simplicity, let's use an environment variable.
os.environ["ORACLE_DATA_DIR"] = str(oracle_data_dir)

# Command to run the Oracle server using Uvicorn
# We need to ensure tsMqlMLOracleServer is importable from the current context
# The server will be run directly from its module path.
server_command = [
    sys.executable, "-m", "uvicorn", "tsMqlMLOracleServer:app",
    "--host", oracle_server_ip,
    "--port", str(oracle_port),
    "--log-level", "info",
    "--reload" # Enable auto-reloading for development
]
# Redirect server output to a specific log file
server_stdout = open(oracle_log_file, "w")
server_stderr = subprocess.STDOUT # Redirect stderr to the same file

logger.info(f"Starting Oracle Server with command: {' '.join(server_command)}")
server_process = subprocess.Popen(server_command, stdout=server_stdout, stderr=server_stderr,
                                  cwd=str(Path(__file__).parent.parent)) # Run from PythonLib directory

# Wait for the Oracle server to start
max_retries = 10
for i in range(max_retries):
    try:
        response = requests.get(f"{oracle_url}/status", timeout=1)
        if response.status_code == 200:
            logger.info("✅ Oracle Server is running.")
            break
    except requests.exceptions.ConnectionError:
        logger.warning(f"Waiting for Oracle Server to start... (Attempt {i+1}/{max_retries})")
        time.sleep(2)
else:
    logger.critical("❌ Oracle Server failed to start after multiple retries. Exiting.")
    if server_process.poll() is not None:
        logger.error(f"Oracle Server process exited with code: {server_process.poll()}")
    sys.exit(1)

# --- Launch Chief and Worker Processes ---
num_workers = tune_params.get('workers', 1) # Default to 1 worker if not specified

child_processes = []

# Launch the chief process
chief_env = os.environ.copy()
chief_env["TUNER_ID"] = "chief"
chief_env["IS_CHIEF"] = "true"
chief_env["BACKEND"] = DEFAULT_BACKEND # Ensure backend is passed
chief_log_file = log_dir / tune_params.get('backend', 'pytorch') / "chief.log"
chief_log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure backend-specific log directory exists
chief_stdout = open(chief_log_file, "w")
chief_stderr = subprocess.STDOUT

logger.info(f"Launching chief process [PID: {os.getpid()}] with backend: {DEFAULT_BACKEND}")
# Pass arguments explicitly to the target function instead of relying solely on environment variables
# This makes the multiprocessing more robust.
chief_process = multiprocessing.Process(
    target=run_chief_process_task,
    args=("chief", oracle_url, True, app_params, tune_params, base_params),
    name="ChiefProcess"
)
chief_process.start()
child_processes.append(chief_process)


# Launch worker processes
for i in range(num_workers - 1): # -1 because one is the chief
    worker_id = f"worker_{i+1}"
    worker_env = os.environ.copy()
    worker_env["TUNER_ID"] = worker_id
    worker_env["IS_CHIEF"] = "false"
    worker_env["BACKEND"] = DEFAULT_BACKEND # Ensure backend is passed
    worker_log_file = log_dir / tune_params.get('backend', 'pytorch') / f"{worker_id}.log"
    worker_log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure backend-specific log directory exists
    worker_stdout = open(worker_log_file, "w")
    worker_stderr = subprocess.STDOUT

    logger.info(f"Launching worker process {worker_id} with backend: {DEFAULT_BACKEND}")
    worker_process = multiprocessing.Process(
        target=run_worker_process_task,
        args=(worker_id, oracle_url, False, app_params, tune_params, base_params),
        name=f"WorkerProcess-{i+1}"
    )
    worker_process.start()
    child_processes.append(worker_process)

logger.info(f"Launched {len(child_processes)} chief/worker processes.")

try:
    while True:
        # Check if all child processes are still alive
        all_finished = True
        for p in child_processes:
            if p.is_alive():
                all_finished = False
                break
        
        if all_finished:
            logger.info("All chief and worker processes have finished.")
            break
        
        # Log status of processes
        for p in child_processes:
            if not p.is_alive():
                logger.warning(f"⚠️ Process {p.name} [PID: {p.pid}] terminated with exit code {p.exitcode}. Check its dedicated log file for details.")
            
        time.sleep(5)

except KeyboardInterrupt:
    logger.info("👋 Launcher received KeyboardInterrupt. Initiating graceful shutdown...")
except Exception as e:
    logger.critical(f"❌ Unhandled exception in launcher: {e}", exc_info=True)
finally:
    logger.info("Terminating all child processes (Chief/Workers)...")
    for p in child_processes:
        if p.is_alive():
            logger.info(f"Terminating process [PID: {p.pid}]...")
            p.terminate()
            try:
                p.join(timeout=10)
            except multiprocessing.TimeoutError:
                logger.warning(f"Process [PID: {p.pid}] did not terminate gracefully, killing it.")
                p.kill()

    logger.info("Terminating Oracle Server process...")
    if server_process.is_alive():
        server_process.terminate()
        try:
            server_process.join(timeout=10)
        except multiprocessing.TimeoutError:
            logger.warning(f"Server process [PID: {server_process.pid}] did not terminate gracefully, killing it.")
            server_process.kill()
    
    logger.info("All processes terminated. Exiting launcher.")
    sys.exit(0)
