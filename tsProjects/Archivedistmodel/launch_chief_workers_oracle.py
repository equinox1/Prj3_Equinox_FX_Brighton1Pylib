#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsLauncherTunerWinMql.py
Description: Launcher script to start chief and multiple workers for tsMqlMLTuner distributed tuning.
Author: Xercescloud
Date: 2025-04-28
Version: 1.0.0
License: MIT
"""

import subprocess
import time
import os
import sys

# ==== CONFIGURATION ====
base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/PythonLib"

CHIEF_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_chief.py")
WORKER_SCRIPT = os.path.join(base_path, "tsProjects/prjNeuroPredict1/tsNeuroPredictWinMql_worker.py")

NUM_WORKERS = 1  # You can adjust this
VENV_PYTHON = os.path.join("venvwin1", "Scripts", "python.exe") if os.name == 'nt' else "python3"

# Environment variables for Oracle Server
os.environ['ORACLE_SERVER_IP'] = "192.168.1.103"
os.environ['ORACLE_SERVER_PORT'] = "9000"

# Helper function to run a process
def run_process(script, role, worker_id=None):
    cmd = [VENV_PYTHON, script]
    env = os.environ.copy()
    env['TUNER_ROLE'] = role
    if worker_id is not None:
        env['WORKER_ID'] = str(worker_id)
    return subprocess.Popen(cmd, env=env)


def main():
    processes = []

    # Start Chief
    print(f"Starting Chief process with {CHIEF_SCRIPT}...")
    chief_proc = run_process(CHIEF_SCRIPT, role="chief")
    processes.append(chief_proc)

    # Wait for Oracle Server to be ready
    print("Waiting 10 seconds for Oracle Server to start...")
    time.sleep(10)

    # Start Workers
    for i in range(NUM_WORKERS):
        print(f"Starting Worker-{i+1} with {WORKER_SCRIPT}...")
        worker_proc = run_process(WORKER_SCRIPT, role="worker", worker_id=i+1)
        processes.append(worker_proc)

    print("All processes launched. Monitoring...")

    try:
        # Wait for all processes
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating all processes...")
        for proc in processes:
            proc.terminate()