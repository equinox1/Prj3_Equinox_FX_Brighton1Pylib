#!/usr/bin/env python3
# +------------------------------------------------------------------+
# | watch_logdir.py (Live Log Monitor)                               |
# +------------------------------------------------------------------+

import os
import time
import threading

# ==== CONFIGURATION ====
log_dir = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir"
log_files = []  # If empty, monitor all logs. Or specify ['chief.log', 'tuner1.log', ...]

poll_interval = 1  # seconds between checks

# ==== FUNCTIONS ====

def follow_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)  # Start at end of file
        while True:
            line = f.readline()
            if line:
                print(f"[{os.path.basename(filepath)}] {line.strip()}")
            else:
                time.sleep(0.5)

def start_monitor():
    print(f"🔍 Monitoring log files in: {log_dir}")

    # Find log files
    if not log_files:
        files_to_watch = [f for f in os.listdir(log_dir) if f.endswith(".log") and not f.endswith("_info.log")]
    else:
        files_to_watch = log_files

    threads = []
    for log_filename in files_to_watch:
        filepath = os.path.join(log_dir, log_filename)
        if os.path.exists(filepath):
            thread = threading.Thread(target=follow_file, args=(filepath,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            print(f"⚠️ Log file not found: {filepath}")

    # Keep main thread alive
    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n❌ Exiting monitor.")

if __name__ == "__main__":
    start_monitor()
