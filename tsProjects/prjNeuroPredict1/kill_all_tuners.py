import psutil
import os
import signal
import platform

SCRIPT_NAMES = [
    "tsNeuroPredictWinMql_chief.py",
    "tsNeuroPredictWinMql_worker.py",
    "tsNeuroPredictWinMql_chief.py",
    "tsNeuroPredictWinMql_worker.py",
    "OracleServer.py",
    "OracleClient.py",
    "uvicorn",
    "winsvrxerces01"
]

def is_windows():
    return platform.system().lower() == "windows"

def kill_by_script_name(script_names):
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline)

            for script_name in script_names:
                if script_name in cmdline_str:
                    print(f"[INFO] Killing PID {proc.pid} | CMD: {cmdline_str}")
                    
                    if is_windows():
                        proc.terminate()
                    else:
                        os.kill(proc.pid, signal.SIGTERM)

                    killed.append((proc.pid, script_name))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed

if __name__ == "__main__":
    killed_procs = kill_by_script_name(SCRIPT_NAMES)
    print(f"[DONE] Killed {len(killed_procs)} matching processes.")
