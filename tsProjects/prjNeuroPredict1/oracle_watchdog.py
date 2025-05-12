import time
import requests
import subprocess
import sys
import os

# === CONFIGURATION ===
ORACLE_URL = "http://192.168.1.103:9000/heartbeat"
TIMEOUT_SECONDS = 30
CHECK_INTERVAL = 1

# Path to OracleServer launch script/module
ORACLE_COMMAND = [
    "uvicorn", "oracle_api:app",  # Change if your app entry point is different
    "--host", "192.168.1.103",          # Important: bind to external IPs
    "--port", "9000"
]
ORACLE_WORKING_DIR = "C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib/tsProjects/prjNeuroPredict1"

CHIEF_COMMAND = [
    "python", "tsNeuroPredictWinMql_chief.py"
]

def start_oracle_server():
    print("[Watchdog] 🚀 Starting OracleServer...")
    subprocess.Popen(
        ORACLE_COMMAND,
        cwd=ORACLE_WORKING_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )

def wait_for_oracle():
    print(f"[Watchdog] Waiting for OracleServer at {ORACLE_URL}...")
    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            response = requests.get(ORACLE_URL, timeout=5)
            if response.status_code == 200:
                print("[Watchdog] ✅ OracleServer is up!")
                return True
        except requests.exceptions.RequestException:
            pass
        print("[Watchdog] OracleServer not ready, retrying...")
        time.sleep(CHECK_INTERVAL)
    print("[Watchdog] ❌ Timeout: OracleServer did not respond in time.")
    return False

def launch_chief():
    print("[Watchdog] 🚀 Launching Chief Tuner...")
    try:
        subprocess.run(CHIEF_COMMAND, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Watchdog] ❌ Chief process failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_oracle_server()
    if wait_for_oracle():
        launch_chief()
    else:
        sys.exit(1)
