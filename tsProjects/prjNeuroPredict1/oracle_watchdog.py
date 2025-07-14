import logging
import warnings
import inspect
import socket
from fastapi import FastAPI
import os
import sys
from pathlib import Path
import subprocess
import requests

# Define scripts paths relative to the launcher script's directory
LAUNCHER_DIR = Path(__file__).parent
ORACLE_SCRIPT = LAUNCHER_DIR / "oracle_server_main.py"
CHIEF_SCRIPT = LAUNCHER_DIR / "tsNeuroPredictWinMql_chief.py"
WORKER_SCRIPT = LAUNCHER_DIR / "tsNeuroPredictWinMql_worker.py"


ORACLE_HOST = "http://192.168.1.103:9000"
CHECK_INTERVAL = 5  # seconds
RESTART_DELAY = 2  # seconds

logging.basicConfig(
    filename="oracle_watchdog.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def is_server_alive():
    try:
        r = requests.get(f"{ORACLE_HOST}/status", timeout=3)
        return r.status_code == 200
    except Exception as e:
        logging.warning(f"Health check failed: {e}")
        return False

def start_server():
    logging.info("Starting Oracle Server...")
    return subprocess.Popen(["python", ORACLE_SCRIPT])

def main():
    proc = None
    try:
        while True:
            if not is_server_alive():
                if proc and proc.poll() is None:
                    logging.info("Terminating unresponsive server...")
                    proc.terminate()
                    time.sleep(RESTART_DELAY)

                proc = start_server()
                time.sleep(RESTART_DELAY)
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Watchdog stopped by user.")
        if proc:
            proc.terminate()

if __name__ == "__main__":
    main()