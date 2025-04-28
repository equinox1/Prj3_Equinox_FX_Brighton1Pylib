# File: launch_chief_workers_parallel.py
import os
import sys
import subprocess
import time
import json
import threading

# ==== CONFIGURATION ====
base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"

chief_script = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
worker_script = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

oracle_ip = '192.168.1.103'
chief_base_port = 8001
worker_base_port = 8002
oracle_server_port = 9000  # <-- Added for Oracle Server
num_workers = 1
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
    full_env.update({str(k): str(v) for k, v in env_vars.items()})
    if tf_config:
        full_env["TF_CONFIG"] = tf_config

    logfile = os.path.join(log_dir, f"{name}.log")
    infofile = os.path.join(log_dir, f"{name}_info.log")
    print(f"📜 Launching {name} -> log: {logfile}")

    with open(logfile, "w") as stdout, open(infofile, "w") as stderr:
        return subprocess.Popen(
            [sys.executable, script],
            env=full_env,
            stdout=stdout,
            stderr=stderr
        )

def threaded_launch(script, env_vars, name, tf_config=None):
    def target():
        launch_process(script, env_vars, name, tf_config)
    thread = threading.Thread(target=target)
    thread.start()
    return thread

# ==== LAUNCH CHIEF FIRST ====

print("🚀 Launching CHIEF...")
chief_env = {
    "TUNER_ID": "chief",
    "ORACLE_SERVER_IP": oracle_ip,
    "ORACLE_SERVER_PORT": str(oracle_server_port)
}
chief_tf_config = build_tf_config("chief", 0, num_workers)
chief_thread = threaded_launch(chief_script, chief_env, "chief", tf_config=chief_tf_config)

time.sleep(2)

# ==== LAUNCH WORKERS ====

print("🧵 Launching WORKERS in parallel...")
worker_threads = []
for i in range(num_workers):
    tuner_id = f"tuner{i+1}"
    port = worker_base_port + i
    worker_env = {
        "TUNER_ID": tuner_id,
        "ORACLE_SERVER_IP": oracle_ip,
        "ORACLE_SERVER_PORT": str(oracle_server_port)
    }
    worker_tf_config = build_tf_config("worker", i, num_workers)
    thread = threaded_launch(worker_script, worker_env, tuner_id, tf_config=worker_tf_config)
    worker_threads.append(thread)

# ==== WAIT FOR ALL THREADS ====

print("\n⏳ Waiting for all workers to launch...")
for thread in worker_threads:
    thread.join(timeout=3)

print(f"\n✅ All tuners launched FAST.\n📂 Logs stored in: {log_dir}")
