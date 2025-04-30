
import os
import signal
import psutil

def kill_by_script_name(script_names):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if any(script in ' '.join(proc.info['cmdline']) for script in script_names):
                print(f"Killing PID {proc.info['pid']} : {' '.join(proc.info['cmdline'])}")
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

if __name__ == "__main__":
    kill_by_script_name(["tsNeuroPredictWinMql_chief.py", "tsNeuroPredictWinMql_worker.py"])
