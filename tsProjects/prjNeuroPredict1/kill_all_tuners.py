import psutil
import os
import signal
import platform
import socket

SCRIPT_NAMES = [
    "tsNeuroPredictWinMql_chief.py",
    "tsNeuroPredictWinMql_worker.py",
    "OracleServer.py",
    "OracleClient.py",
    "uvicorn",
    "winsvrxerces01",
]

TARGET_PORT = 9000

def is_windows():
    return platform.system().lower() == "windows"

def kill_process(proc):
    try:
        if is_windows():
            proc.terminate()
        else:
            os.kill(proc.pid, signal.SIGTERM)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

def kill_by_script_name(script_names):
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline).lower()

            for script_name in script_names:
                if script_name.lower() in cmdline_str:
                    print(f"[INFO] Killing PID {proc.pid} | CMD: {cmdline_str}")
                    if kill_process(proc):
                        killed.append((proc.pid, script_name))
                    break
        except Exception:
            continue
    return killed

def kill_by_port(port):
    killed = []
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                proc = psutil.Process(conn.pid)
                print(f"[INFO] Killing PID {proc.pid} using port {port}")
                if kill_process(proc):
                    killed.append((proc.pid, f"port {port}"))
            except Exception:
                continue
    return killed
#

if __name__ == "__main__":
    killed_procs = kill_by_script_name(SCRIPT_NAMES)
    killed_by_port = kill_by_port(TARGET_PORT)
    total_killed = killed_procs + killed_by_port

    print(f"[DONE] Killed {len(total_killed)} total matching processes.")
