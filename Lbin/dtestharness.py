import subprocess
import os
import time
import signal
import json
from pathlib import Path
import threading

# ----------- Configurable Paths ----------
ORACLE_SERVER = "oracle_server_main.py"
WORKER_SCRIPT = "tsNeuroPredictWinMql_worker.py"
ORACLE_DIR = Path("Logdir/oracle_server")  # Matches your config
ORACLE_JSON = ORACLE_DIR / "oracle.json"
WAIT_SECONDS = 30

# ----------- Environment Vars ----------
os.environ["TUNER_ID"] = "test_worker"
os.environ["BACKEND"] = "pytorch"
os.environ["ML_MODEL_NAME"] = "test_model"
os.environ["ML_PROJECT_ID"] = "123"
os.environ["ORACLE_SERVER_HOST"] = "127.0.0.1"
os.environ["ORACLE_SERVER_PORT"] = "9000"

def stream_output(prefix, stream):
    for line in iter(stream.readline, b''):
        print(f"{prefix} {line.decode().rstrip()}")
    stream.close()

# ----------- Launch Oracle Server ----------
print("[TEST] Launching Oracle Server...")
oracle_proc = subprocess.Popen(
    ["python", ORACLE_SERVER],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
threading.Thread(target=stream_output, args=("[ORACLE]", oracle_proc.stdout), daemon=True).start()
threading.Thread(target=stream_output, args=("[ORACLE-ERR]", oracle_proc.stderr), daemon=True).start()

time.sleep(5)  # Give the server time to start

# ----------- Launch Worker ----------
print("[TEST] Launching Worker...")
worker_proc = subprocess.Popen(
    ["python", WORKER_SCRIPT],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
threading.Thread(target=stream_output, args=("[WORKER]", worker_proc.stdout), daemon=True).start()
threading.Thread(target=stream_output, args=("[WORKER-ERR]", worker_proc.stderr), daemon=True).start()

# ----------- Wait for Activity ----------
print(f"[TEST] Waiting up to {WAIT_SECONDS}s for worker activity...")
time.sleep(WAIT_SECONDS)

# ----------- Validate oracle.json ----------
if ORACLE_JSON.exists():
    print(f"[TEST] ✅ Found oracle.json at {ORACLE_JSON}")
    try:
        with open(ORACLE_JSON, "r") as f:
            data = json.load(f)
        trials = data.get("trials", {})
        if trials:
            print(f"[TEST] ✅ {len(trials)} trial(s) recorded in oracle.json")
        else:
            print("[TEST] ❌ oracle.json found, but no trials recorded.")
    except Exception as e:
        print(f"[TEST] ❌ Failed to read oracle.json: {e}")
else:
    print("[TEST] ❌ oracle.json not found.")

# ----------- Cleanup ----------
print("[TEST] Cleaning up processes...")
for proc in [worker_proc, oracle_proc]:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()

print("[TEST] ✅ Done.")
