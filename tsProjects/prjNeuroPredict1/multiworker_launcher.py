import subprocess
import time
import os
import socket
import requests
import sys
import psutil
import logging

from tsMqlSetup import CMqlSetup

# ==== CONFIGURATION ====
NUM_WORKERS = 1  # Number of worker processes to launch
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
GLOBAL_TUNER_TYPE = "hyperband"  # <-- Ensures Oracle and Chief use the same tuner type


# ==== UTILITIES ====
def port_in_use(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (OSError, socket.timeout):
        return False

def kill_process_on_port(port):
    print(f"\U0001F50D Scanning for processes using port {port}...")
    for proc in psutil.process_iter(['pid', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    print(f"\U0001F52A Killing process {proc.pid} using port {port}...")
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
    env["TUNER_TYPE"] = GLOBAL_TUNER_TYPE  # <-- Ensures consistent tuner mode
    


    label = tuner_id.upper() if tuner_id else "PROCESS"
    print(f"[LAUNCH] Launching {label} → {script_path}")
    return subprocess.Popen([PYTHON_EXEC, script_path], env=env)

def wait_for_oracle_ready():
    print(f"\u23F3 Waiting for OracleServer at {ORACLE_URL}...")
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

    print("🧑‍🔬 Launching Worker Processes...")
    workers = [launch_process(WORKER_SCRIPT, tuner_id=f"worker{i+1}", backend=GLOBAL_BACKEND) for i in range(NUM_WORKERS)]

    try:
        chief_proc.wait()
    except KeyboardInterrupt:
        print("🚫 KeyboardInterrupt received. Shutting down processes...")
    finally:
        if chief_proc:
            chief_proc.terminate()
        for worker in workers:
            worker.terminate()
        if oracle_proc:
            oracle_proc.terminate()
