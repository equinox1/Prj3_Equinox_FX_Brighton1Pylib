
import subprocess
import time
import os


NUM_WORKERS = 1
PYTHON_EXEC = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\venvwin1\Scripts\python.exe"
# ==== CONFIGURATION ====

base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"
CHIEF_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
WORKER_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

def launch_process(script_name, tuner_id):
    env = os.environ.copy()
    env["TUNER_ID"] = tuner_id
    return subprocess.Popen([PYTHON_EXEC, script_name], env=env)

if __name__ == "__main__":
    print("Starting Chief...")
    chief_proc = launch_process(CHIEF_SCRIPT, "chief")
    time.sleep(5)  # give the chief time to start the OracleServer

    workers = []
    for i in range(NUM_WORKERS):
        print(f"Starting Worker-{i+1}...")
        proc = launch_process(WORKER_SCRIPT, f"worker_{i+1}")
        workers.append(proc)

    try:
        chief_proc.wait()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        chief_proc.terminate()
        for w in workers:
            w.terminate()
