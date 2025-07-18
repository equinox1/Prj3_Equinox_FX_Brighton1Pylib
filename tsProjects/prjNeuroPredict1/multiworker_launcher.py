# filename: multiworker_launcher.py
# Rewritten: patched_oracle_server_main.py
runtuner= 'pt' # 'pt' for PyTorch, 'tf' for TensorFlow
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

# --- PATH CONFIGURATION FOR MODULE IMPORTS ---
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

# Network config
xerces_server = app_params.get('xerces_server', "127.0.0.1")
xerces_port = app_params.get('xerces_port', 9000)
print(f"Using Xerces server: {xerces_server}, port: {xerces_port}")

# --- Backend Selection Variable ---
# Define the default backend here. Change this variable to switch between backends.
# Check for an environment variable 'ML_BACKEND' first, otherwise default to 'tensorflow'.
if runtuner == 'pt':
    DEFAULT_BACKEND = 'pytorch'
else:
    # Default to 'tensorflow' if not specified
    # This allows the user to set an environment variable to override the default backend.
    # If the environment variable is not set, it will default to 'tensorflow'.
    # This is useful for users who want to run the code with a specific backend without modifying
    # the source code.
    DEFAULT_BACKEND = os.environ.get('ML_BACKEND', 'tensorflow').lower() # Options: 'tensorflow', 'pytorch'

# Validate the selected backend
if DEFAULT_BACKEND not in ['tensorflow', 'pytorch']:
    logger.error(f"Invalid ML_BACKEND environment variable value: {DEFAULT_BACKEND}. Defaulting to 'tensorflow'.")
    DEFAULT_BACKEND = 'tensorflow'

# Override backend in tune_params with the defined variable
tune_params['backend'] = DEFAULT_BACKEND
logger.info(f"Selected backend: {DEFAULT_BACKEND} (from ML_BACKEND env variable or default)")

# Set the number of epochs and trials as requested
tune_params['max_epochs'] = 100
tune_params['num_trials'] = 50
logger.info(f"Set max_epochs to {tune_params['max_epochs']} and num_trials to {tune_params['num_trials']}")


# Suppress deprecated warnings (if any)
warnings.filterwarnings("ignore", message="The `tune_new_entries` and `allow_new_entries` arguments are deprecated.")

# Directory setup
import random
oracle_base_dir = Path(base_params.get('mp_glob_base_log_path')) / "oracle_server_data"
oracle_base_dir.mkdir(parents=True, exist_ok=True)

model_name = os.environ.get('ML_MODEL_NAME', tune_params.get('ml_model_name', 'default_model'))
project_id = os.environ.get('ML_PROJECT_ID', str(base_params.get('mp_glob_sub_ml_baseuniq', random.randrange(1, 1024))))
project_name = f"{model_name}_{project_id}"

oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Oracle data directory: {oracle_full_path}")


def run_oracle_server(host: str, port: int, oracle_base_dir: Path, project_name: str, tune_params: dict):
    """
    Function to run the Oracle Server, to be executed in a separate process.
    It imports and initializes the Oracle and FastAPI app within its own process space.
    """
    import logging
    from pathlib import Path
    import uvicorn
    import socket
    from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
    from tsMqlMLTuner.tsMqlMLOracleServer import app as oracle_app_instance, set_oracle_instance

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    oracle_logger = logging.getLogger("OracleServerProcess")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            s.listen(1)
            oracle_logger.info(f"Port {port} is available for Oracle Server.")
        except socket.error as e:
            oracle_logger.critical(f"Port {port} is already in use or cannot be bound by Oracle Server: {e}. Exiting server process.")
            sys.exit(1)


    oracle_instance = CustomOracle(
        objective=tune_params.get('objective', "val_loss"),
        max_trials=tune_params.get('num_trials', 50),
        directory=str(oracle_base_dir),
        project_name=project_name,
        seed=tune_params.get('seed', 42),
        overwrite=tune_params.get('overwrite', False)
    )
    oracle_logger.info("CustomOracle instance created.")

    set_oracle_instance(oracle_instance)
    
    oracle_logger.info(f"✅ Attempting to start Oracle Server at http://{host}:{port}")

    try:
        uvicorn.run(oracle_app_instance, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        oracle_logger.info("👋 Oracle Server received KeyboardInterrupt. Shutting down.")
    except Exception as e:
        oracle_logger.critical(f"❌ Oracle Server Uvicorn process failed: {e}", exc_info=True)
        (oracle_base_dir / "server_startup_failed.flag").touch()
    finally:
        oracle_logger.info("Oracle Server process terminated.")

def check_server_status(url: str, timeout: int = 5) -> bool:
    """Checks if the FastAPI server is up and running."""
    try:
        response = requests.get(f"{url}/status", timeout=timeout)
        response.raise_for_status()
        logger.info(f"Oracle Server is available at {url}. Status: {response.json()}")
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
        logger.warning(f"Oracle Server not yet available at {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when checking Oracle Server status: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import asyncio
    import platform

    # Removed incorrect asyncio.set_event_loop_policy call
    # On Windows, multiprocessing defaults to 'spawn' which is usually fine.
    # If explicit setting is ever needed, it's multiprocessing.set_start_method('spawn', force=True)
    # but it's generally not required unless specific issues arise.

    server_url = f"http://{xerces_server}:{xerces_port}"

    # Start Oracle Server in a separate process
    logger.info("🚀 Starting Oracle Server process...")
    server_process = multiprocessing.Process(
        target=run_oracle_server,
        args=(xerces_server, xerces_port, oracle_full_path, project_name, tune_params)
    )
    server_process.start()

    logger.info("Waiting for Oracle Server to start...")
    max_wait_time = 60
    start_time = time.time()
    server_ready = False
    while time.time() - start_time < max_wait_time:
        if check_server_status(server_url):
            server_ready = True
            break
        if (oracle_full_path / "server_startup_failed.flag").exists():
            logger.critical("Oracle Server startup failed as indicated by flag file. Aborting launcher.")
            server_process.terminate()
            server_process.join()
            sys.exit(1)
        time.sleep(2)

    if not server_ready:
        logger.critical("Oracle Server did not start in time. Aborting multi-worker training.")
        server_process.terminate()
        server_process.join()
        sys.exit(1)

    logger.info("Oracle Server is up and running. Proceeding with Chief and Worker processes.")

    num_additional_workers = tune_params.get('num_workers', 0)
    
    # List of target functions for multiprocessing
    worker_tasks = [run_chief_process_task] + [run_worker_process_task] * num_additional_workers

    child_processes = []

    try:
        # Create a base directory for all logs if it doesn't exist
        # Note: With multiprocessing.Process, logs are handled by tsMqlLogService
        # directly writing to files, so stdout/stderr redirection is not needed here.
        # However, the multiworker_logs directory might still be useful for other purposes
        # or if you want a fallback capture.
        multiworker_log_dir = Path(base_params.get('mp_glob_base_log_path')) / "multiworker_logs"
        multiworker_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Multi-worker logs (managed by tsMqlLogService) will be in: {multiworker_log_dir.parent}")


        for i, task_function in enumerate(worker_tasks):
            process_type = "chief" if i == 0 else f"worker_{i}"
            is_current_chief = (i == 0)

            # Pass all necessary parameters as arguments to the target function
            # Environment variables are still set for backward compatibility/redundancy,
            # but direct argument passing is preferred for multiprocessing.
            args = (
                process_type,
                server_url,
                is_current_chief,
                app_params,
                tune_params,
                base_params
            )

            logger.info(f"Starting {process_type} process via multiprocessing.Process...")
            p = multiprocessing.Process(
                target=task_function,
                args=args,
                # No stdout/stderr redirection here; tsMqlLogService handles file logging.
                # No cwd needed here; the target function's module will be imported.
            )
            child_processes.append(p)
            p.start() # Start the process
            time.sleep(1) # Give a moment for the process to start

        # Monitor child processes
        while True:
            all_finished = True
            for i, p in enumerate(child_processes):
                if p.is_alive(): # Check if process is still running
                    all_finished = False
                elif p.exitcode != 0: # Process terminated with an error
                    process_type = "Chief" if i == 0 else f"Worker {i}"
                    logger.warning(f"⚠️ {process_type} process [PID: {p.pid}] terminated with exit code {p.exitcode}. Check its dedicated log file for details.")
            
            if all_finished:
                logger.info("All chief and worker processes have finished.")
                break
            
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
