import os
import sys
import subprocess
import time
import socket
import json

# ==== CONFIGURATION ====
base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"

chief_script = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
worker_script = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

oracle_ip = '192.168.1.103'  # Replace with your actual IP
oracle_port = '8000'
chief_base_port = 8001
worker_base_port = 8002
num_workers = 48
log_dir = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir"

os.makedirs(log_dir, exist_ok=True)

# ==== FUNCTIONS ====

def build_tf_config(role, index, num_workers):
    cluster = {
        "chief": [f"{oracle_ip}:{chief_base_port}"],
        "worker": [f"{oracle_ip}:{worker_base_port + i}" for i in range(num_workers)]
    }
    return json.dumps({
        "cluster": cluster,
        "task": {"type": role, "index": index}
    })

def launch_process(script, env_vars, name, tf_config=None):
    full_env = os.environ.copy()
    # Convert all environment values to strings
    full_env.update({str(k): str(v) for k, v in env_vars.items()})
    if tf_config:
        full_env["TF_CONFIG"] = str(tf_config)

    logfile = os.path.join(log_dir, f"{name}.log")
    errfile = os.path.join(log_dir, f"{name}_err.log")
    print(f"📜 Logging to: {logfile} and {errfile}")
    print(f"system: {sys.executable}")
    print(f"script: {script}")
    print(f"TF_CONFIG: {tf_config}")

    with open(logfile, "w") as stdout, open(errfile, "w") as stderr:
        return subprocess.Popen(
            [sys.executable, script],
            env=full_env,
            stdout=stdout,
            stderr=stderr
        )

def wait_for_port(ip, port, timeout=45):
    print(f"\u23f3 Waiting for Oracle at {ip}:{port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((ip, int(port)), timeout=2):
                print(f"✅ Oracle available at {ip}:{port}")
                return True
        except Exception:
            time.sleep(1)
    print(f"❌ Timed out waiting for Oracle at {ip}:{port}")
    return False

# ==== LAUNCH CHIEF ====

print("🚀 Launching CHIEF...")
chief_env = {
    "TUNER_ID": "chief",
    "KERASTUNER_TUNER_ID": "chief",
    "KERASTUNER_ORACLE_IP": oracle_ip,
    "KERASTUNER_ORACLE_PORT": oracle_port,
    "KERASTUNER_ORACLE_WORKER": "True",
    "KERASTUNER_ORACLE_WORKER_ID": "chief_worker",
    "KERASTUNER_ORACLE_WORKER_PORT": str(chief_base_port),
}

chief_tf_config = build_tf_config("chief", 0, num_workers)
chief_proc = launch_process(chief_script, chief_env, "chief", tf_config=chief_tf_config)
print(f"chief_proc: {chief_proc}")
# ==== WAIT FOR ORACLE TO BECOME AVAILABLE ====

if not wait_for_port(oracle_ip, oracle_port, timeout=30):
    chief_proc.terminate()
    raise SystemExit("❌ Chief failed to start. Aborting worker launch.")

# ==== LAUNCH WORKERS ====

workers = []
for i in range(num_workers):
    tuner_id = f"tuner{i+1}"
    port = str(worker_base_port + i)

    print(f"🧵 Launching WORKER {tuner_id} on port {port}...")

    worker_env = {
        "TUNER_ID": tuner_id,
        "KERASTUNER_TUNER_ID": tuner_id,
        "KERASTUNER_ORACLE_IP": oracle_ip,
        "KERASTUNER_ORACLE_PORT": oracle_port,
        "KERASTUNER_ORACLE_WORKER": "True",
        "KERASTUNER_ORACLE_WORKER_ID": tuner_id,
        "KERASTUNER_ORACLE_WORKER_PORT": port,
    }

    worker_tf_config = build_tf_config("worker", i, num_workers)
    worker_proc = launch_process(worker_script, worker_env, tuner_id, tf_config=worker_tf_config)
    workers.append((tuner_id, worker_proc))

print(f"\n✅ All tuners launched.\n📂 Logs stored in: {log_dir}")
