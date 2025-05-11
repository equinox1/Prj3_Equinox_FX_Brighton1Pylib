import subprocess
import time
import os
import socket
import requests
import sys
import psutil

NUM_WORKERS = 1
PYTHON_EXEC = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\venvwin1\Scripts\python.exe"

# ==== CONFIGURATION ====
base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"
CHIEF_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
WORKER_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")
ORACLE_HOST = '192.168.1.103'
ORACLE_PORT = 9000
ORACLE_URL = f"http://{ORACLE_HOST}:{ORACLE_PORT}"
MAX_WAIT_SECONDS = 60
FORCE_KILL = '--force' in sys.argv

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

def launch_process(script_path, tuner_id):
    env = os.environ.copy()
    env["TUNER_ID"] = tuner_id
    return subprocess.Popen([PYTHON_EXEC, script_path], env=env)

def wait_for_oracle_ready(timeout=MAX_WAIT_SECONDS):
    print(f"⏳ Waiting for OracleServer to start on {ORACLE_URL}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((ORACLE_HOST, ORACLE_PORT), timeout=2):
                resp = requests.get(f"{ORACLE_URL}/get_trial", timeout=2)
                if resp.status_code == 200:
                    print("✅ OracleServer is online and /get_trial is responsive.")
                    return True
        except Exception as e:
            print(f"[WAIT] Oracle not ready yet: {e}")
            time.sleep(2)
    print("❌ ERROR: OracleServer failed to start within timeout.")
    return False

def is_oracle_already_running():
    try:
        with socket.create_connection((ORACLE_HOST, ORACLE_PORT), timeout=2):
            print(f"🔄 OracleServer already running at {ORACLE_HOST}:{ORACLE_PORT}. Skipping chief.")
            return True
    except Exception:
        return False

if __name__ == "__main__":
    if port_in_use(ORACLE_HOST, ORACLE_PORT):
        if FORCE_KILL:
            print(f"⚠️ Port {ORACLE_PORT} in use. Attempting forced shutdown...")
            kill_process_on_port(ORACLE_PORT)
            time.sleep(2)
        elif is_oracle_already_running():
            chief_proc = None
        else:
            print(f"⚠️ Port {ORACLE_PORT} is already in use. OracleServer may already be running.")
            chief_proc = None
    else:
        print("🚀 Starting OracleServer Chief...")
        chief_proc = launch_process(CHIEF_SCRIPT, "chief")
        if not wait_for_oracle_ready():
            print("❌ Exiting: OracleServer not responsive.")
            if chief_proc:
                chief_proc.terminate()
                chief_proc.wait()
            sys.exit(1)

    time.sleep(2)

    print("🧑‍🏭 Starting Worker(s)...")
    workers = []
    for i in range(NUM_WORKERS):
        print(f"🟢 Launching Worker-{i+1}")
        proc = launch_process(WORKER_SCRIPT, f"worker_{i+1}")
        workers.append(proc)

    try:
        if 'chief_proc' in locals() and chief_proc:
            chief_proc.wait()
        else:
            while True:
                time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Stopping all processes...")
        if 'chief_proc' in locals() and chief_proc:
            chief_proc.terminate()
        for w in workers:
            w.terminate()
