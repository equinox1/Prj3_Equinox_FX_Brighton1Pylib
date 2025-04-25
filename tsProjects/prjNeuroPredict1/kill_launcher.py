#!/usr/bin/env python3
# +------------------------------------------------------------------+
# | kill_launcher.py (Stop Chief + Workers cleanly)                  |
# +------------------------------------------------------------------+

import os
import subprocess
import time
import signal

# ==== CONFIGURATION ====
# Same as launcher
log_dir = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir"

# These are the *names* of the processes to look for
process_names = [
    "tsNeuroPredictWinMql_chief.py",
    "tsNeuroPredictWinMql_worker.py"
]

# Python executable name
python_executable = "python.exe"  # or "python" if running on Linux

# ==== FUNCTIONS ====

def find_processes():
    """Find all PIDs of matching Python processes running chief/worker scripts."""
    matching_pids = []

    try:
        # Run 'tasklist' on Windows to find all running processes
        result = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {python_executable}"], capture_output=True, text=True)
        processes = result.stdout.splitlines()

        for line in processes:
            if any(name in line for name in process_names):
                # Extract PID from line
                parts = line.split()
                if len(parts) > 1:
                    pid = int(parts[1])
                    matching_pids.append(pid)
    except Exception as e:
        print(f"❌ Error finding processes: {e}")

    return matching_pids

def kill_processes(pids):
    """Send SIGTERM to all matching PIDs."""
    for pid in pids:
        try:
            print(f"🔪 Killing PID {pid}...")
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            print(f"❌ Failed to kill PID {pid}: {e}")

# ==== MAIN ====

def main():
    print("🔍 Finding launcher processes...")
    pids = find_processes()

    if not pids:
        print("✅ No matching chief or worker processes found.")
        return

    print(f"❗ Found {len(pids)} matching processes.")
    kill_processes(pids)

    print("✅ All processes sent SIGTERM. Give them a few seconds to shut down.")
    time.sleep(2)

if __name__ == "__main__":
    main()
