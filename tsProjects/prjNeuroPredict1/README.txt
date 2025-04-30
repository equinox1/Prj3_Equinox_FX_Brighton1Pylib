
Tuning Utility Scripts for TensorFlow/Keras Distributed ML System
=================================================================

Included Scripts:
-----------------

1. multiworker_launcher.py
   - Launches the 'chief' and multiple 'worker' processes for distributed tuning.
   - Set the number of workers by modifying the NUM_WORKERS variable.
   - Example usage:
       $ python multiworker_launcher.py

2. kill_all_tuners.py
   - Kills all processes running chief or worker scripts.
   - Uses psutil to find and terminate processes by name.
   - Example usage:
       $ python kill_all_tuners.py

3. log_dashboard.py
   - FastAPI-based web dashboard to view the latest logs.
   - Opens a browser view at: http://localhost:8080
   - Example usage:
       $ python log_dashboard.py

Prerequisites:
--------------
- Python 3.8+
- FastAPI, uvicorn, psutil (install via pip if missing)
    $ pip install fastapi uvicorn psutil

Notes:
------
- Ensure all scripts (chief/worker) and logs are in the same working directory.
- Log output file should be 'tsneuropredict_app.log' or adjust in `log_dashboard.py`.

(c) XercesCloud — Tony Shepherd
