import subprocess
import time
import os
import socket
import requests
import sys

NUM_WORKERS = 2
PYTHON_EXEC = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\venvwin1\Scripts\python.exe"

# ==== CONFIGURATION ====
base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"
CHIEF_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/oracle_server_main.py")
WORKER_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")
ORACLE_HOST = '192.168.1.103'
ORACLE_PORT = 9000
ORACLE_URL = f"http://{ORACLE_HOST}:{ORACLE_PORT}"
MAX_WAIT_SECONDS = 60

def port_in_use(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (OSError, socket.timeout):
        return False

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

if __name__ == "__main__":
    if port_in_use(ORACLE_HOST, ORACLE_PORT):
        print(f"⚠️ Port {ORACLE_PORT} is already in use. OracleServer may already be running.")
        print("🔄 Skipping Chief launch.")
    else:
        print("🚀 Starting OracleServer Chief...")
        chief_proc = launch_process(CHIEF_SCRIPT, "chief")
        if not wait_for_oracle_ready():
            print("❌ Exiting: OracleServer not responsive.")
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
        if 'chief_proc' in locals():
            chief_proc.wait()
        else:
            while True:
                time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Stopping all processes...")
        if 'chief_proc' in locals():
            chief_proc.terminate()
        for w in workers:
            w.terminate()
