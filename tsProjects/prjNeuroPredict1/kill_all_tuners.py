import psutil
import os
import signal

# List of script names to kill (customize this list as needed)
SCRIPT_NAMES = [
    "chief_script.py",
    "worker_script.py",
    "oracle_server.py",
    "oracle_client.py"
]

def kill_by_script_name(script_names):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline)
            for script_name in script_names:
                if script_name in cmdline_str:
                    print(f"Killing PID {proc.pid} with command line: {cmdline_str}")
                    os.kill(proc.pid, signal.SIGTERM)
                    break  # No need to check other script names for this process
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  # Skip processes that no longer exist or can't be accessed

if __name__ == "__main__":
    kill_by_script_name(SCRIPT_NAMES)
