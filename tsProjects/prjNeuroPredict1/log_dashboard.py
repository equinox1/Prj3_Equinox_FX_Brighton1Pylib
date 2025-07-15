from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import html
import glob
import json
from datetime import datetime
from pathlib import Path # Import Path for robust path handling

# Import configuration modules to dynamically determine log path
from tsMqlOverrides import CMqlOverrides
from tsMqlLogService import CMLogServiceSetup # To understand log path structure

# Load configuration (similar to other main scripts)
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

# Determine the backend from environment, default to 'pytorch'
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

# Initialize logging for the dashboard itself, using the same setup principles
# This ensures the dashboard's own logs go to the correct place if needed,
# but primarily we need it to understand the structure for tsneuropredict_app.log
dashboard_logger = CMLogServiceSetup.initialize_logging(
    app_params=app_params,
    tune_params=tune_params,
    base_params=base_params,
    role_hint=__name__,
    loglevel='INFO',
    logfile='log_dashboard.log', # A separate log file for the dashboard itself
    # Removed 'backend=backend_for_log' as it's not expected by CMLogServiceSetup.initialize_logging
)

# Dynamically resolve log file
def resolve_logfile():
    # Retrieve the base log directory configured in app_params
    # This should match the LOGDIR set in multiworker_launcher and chief/oracle
    base_log_dir = Path(app_params.get('LOGDIR', 'Logdir'))
    
    # Construct the expected path based on backend and logfile name
    # The CMLogServiceSetup uses a structure like: base_log_dir / backend / tsneuropredict_app.log
    expected_log_file_path = base_log_dir / backend_for_log / 'tsneuropredict_app.log'
    
    dashboard_logger.info(f"Attempting to resolve log file at: {expected_log_file_path.resolve()}")

    if expected_log_file_path.exists():
        dashboard_logger.info(f"Resolved log file to: {expected_log_file_path.resolve()}")
        return str(expected_log_file_path.resolve()) # Return as string
    else:
        # Fallback: search recursively if direct path doesn't exist (less efficient but robust)
        dashboard_logger.warning(f"Expected log file '{expected_log_file_path}' not found. Attempting recursive search.")
        # Recursively search within the base_log_dir for the target log file
        matches = glob.glob(os.path.join(str(base_log_dir), "**", "tsneuropredict_app.log"), recursive=True)
        if matches:
            # Return the most recently modified log file if multiple found
            most_recent_log = max(matches, key=os.path.getmtime)
            dashboard_logger.info(f"Found log file via recursive search: {most_recent_log}")
            return most_recent_log
        dashboard_logger.error("Could not find 'tsneuropredict_app.log' via direct path or recursive search.")
    return None

LOG_FILE = resolve_logfile()
# Oracle API address and port, fetched from environment variable or default
# Use ORACLE_SERVER_HOST and ORACLE_SERVER_PORT from app_params, falling back to env/defaults
ORACLE_API_HOST = app_params.get('xerces_server', os.getenv("ORACLE_SERVER_HOST", "192.168.1.103"))
ORACLE_API_PORT = app_params.get('xerces_port', int(os.getenv("ORACLE_SERVER_PORT", 9000)))
ORACLE_API = f"http://{ORACLE_API_HOST}:{ORACLE_API_PORT}"
dashboard_logger.info(f"Oracle API endpoint set to: {ORACLE_API}")


app = FastAPI(title="Tuner Dashboard")

# Configure CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def html_template(title: str, body: str, extra_scripts: str = "") -> str:
    """
    Generates the basic HTML structure for the dashboard pages.
    """
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <meta id='refresh-meta' http-equiv="refresh" content="10">
            <script>
            // Toggles the display of an element by its ID
            function toggle(id) {{
                const el = document.getElementById(id);
                el.style.display = el.style.display === 'none' ? 'block' : 'none';
            }}
            // Filters table rows based on selected status
            function filterTable() {{
                const filter = document.getElementById('statusFilter').value;
                const rows = document.querySelectorAll("table tr");
                rows.forEach((row, index) => {{
                    if (index === 0) return; // Skip header row
                    const statusCell = row.cells[1]; // Status is in the second column
                    const show = !filter || statusCell.textContent.trim() === filter;
                    row.style.display = show ? "" : "none";
                }});
            }}
            // Copies text content of an element to the clipboard
            function copyToClipboard(id) {{
                const el = document.getElementById(id);
                // Using document.execCommand('copy') for better compatibility within iframes
                const range = document.createRange();
                range.selectNode(el);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                document.execCommand('copy');
                window.getSelection().removeAllRanges();
                // Optionally provide user feedback that text has been copied
                // A more robust UI would use a temporary message box instead of alert
                // alert('Copied to clipboard!');
            }}
            // Toggles auto-refreshing of the page
            function toggleRefresh() {{
                const meta = document.getElementById('refresh-meta');
                meta.content = document.getElementById('autorefresh').checked ? "10" : "";
            }}
            {extra_scripts}
            </script>
            <style>
                body {{ font-family: 'Inter', monospace; padding: 20px; background-color: #f4f7f6; color: #333; }}
                .container {{ max-width: 1200px; margin: 20px auto; background-color: #fff; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 30px; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
                h2 {{ color: #2c3e50; margin-top: 0; }}
                .logbox {{ white-space: pre-wrap; background: #eef1f3; border: 1px solid #ccc; padding: 15px; height: 600px; overflow-y: scroll; border-radius: 8px; font-size: 0.9em; line-height: 1.4; }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 20px; border-radius: 8px; overflow: hidden; }}
                th, td {{ padding: 12px 15px; border-bottom: 1px solid #eee; text-align: left; }}
                th {{ background-color: #e0e6ea; color: #34495e; font-weight: bold; text-transform: uppercase; font-size: 0.85em; }}
                tr:last-child td {{ border-bottom: none; }}
                tr:hover {{ background-color: #f5f5f5; }}
                a {{ display: inline-block; margin-top: 20px; color: #3498db; text-decoration: none; font-weight: bold; }}
                a:hover {{ text-decoration: underline; }}
                .status-RUNNING {{ color: #f39c12; font-weight: bold; }}
                .status-COMPLETED {{ color: #27ae60; font-weight: bold; }}
                .status-FAILED {{ color: #e74c3c; font-weight: bold; }}
                .toggle-btn {{ cursor: pointer; color: #3498db; text-decoration: underline; margin-right: 8px; }}
                .best-trial {{ background-color: #dff0d8 !important; border-left: 5px solid #27ae60; }}
                .error-message {{ color: #e74c3c; background-color: #fbecec; border: 1px solid #e74c3c; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .progress-bar-container {{ width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 10px; }}
                .progress-bar {{ height: 20px; background-color: #3498db; border-radius: 5px; text-align: center; color: white; line-height: 20px; font-size: 0.8em; }}
                button {{ background-color: #007bff; color: white; border: none; border-radius: 5px; padding: 8px 12px; cursor: pointer; font-size: 0.8em; }}
                button:hover {{ background-color: #0056b3; }}
                select, input[type="checkbox"] {{ margin-right: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <label><input type="checkbox" id="autorefresh" checked onchange="toggleRefresh()"> Auto-refresh</label>
                </div>
                {body}
            </div>
        </body>
    </html>
    """

def safe_str(obj):
    """Safely converts an object to a string, handling potential encoding issues."""
    try:
        return str(obj)
    except Exception:
        return repr(obj)

@app.get("/", response_class=HTMLResponse)
def show_logs():
    """
    Displays the live logs from the tsneuropredict_app.log file.
    """
    if not LOG_FILE or not Path(LOG_FILE).exists(): # Use Path.exists() for robustness
        return HTMLResponse(html_template("Log Viewer", "<div class='error-message'><h3>Error: Log file not found.</h3><p>Please ensure 'tsneuropredict_app.log' exists in the configured log directory.</p></div>"), status_code=404)

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors='ignore') as f:
            lines = f.readlines()[-300:] # Get the last 300 lines for live view
    except Exception as e:
        return HTMLResponse(html_template("Log Viewer", f"<div class='error-message'><h3>Error reading log file:</h3><p>{html.escape(str(e))}</p></div>"), status_code=500)

    escaped_log = html.escape("".join(lines))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = f"<h2>Tuning Logs (Live)</h2><p><em>Last updated: {timestamp}</em></p><div class='logbox'>{escaped_log}</div><a href='/trials'>→ View Trials Dashboard</a>"
    return HTMLResponse(html_template("Log Viewer", body))

@app.get("/trials", response_class=HTMLResponse)
def show_trials():
    """
    Displays a dashboard of tuning trials fetched from the Oracle server.
    Includes summary, progress, and detailed trial information.
    """
    oracle_status = {"max_trials": "N/A", "objective": "N/A", "active_trials": "N/A"}
    oracle_api_reachable = True
    
    # Try to fetch Oracle status
    try:
        status_response = requests.get(f"{ORACLE_API}/status", timeout=5)
        status_response.raise_for_status() # Raise an exception for HTTP errors
        oracle_status = status_response.json()
    except requests.exceptions.ConnectionError:
        oracle_api_reachable = False
        dashboard_logger.error(f"Failed to connect to Oracle API at {ORACLE_API}. Is the server running?")
    except requests.exceptions.Timeout:
        oracle_api_reachable = False
        dashboard_logger.error(f"Timeout connecting to Oracle API at {ORACLE_API}.")
    except Exception as e:
        oracle_api_reachable = False
        dashboard_logger.error(f"Error fetching Oracle status from {ORACLE_API}/status: {e}", exc_info=True)

    trials = []
    # Try to fetch trials list
    if oracle_api_reachable:
        try:
            response = requests.get(f"{ORACLE_API}/list_trials", timeout=10) # Increased timeout
            response.raise_for_status()
            trials = response.json().get("trials", [])
        except requests.exceptions.ConnectionError:
            oracle_api_reachable = False # Mark as unreachable if trials fetch fails too
            dashboard_logger.error(f"Failed to connect to Oracle API at {ORACLE_API} when fetching trials.")
        except requests.exceptions.Timeout:
            oracle_api_reachable = False
            dashboard_logger.error(f"Timeout connecting to Oracle API at {ORACLE_API} when fetching trials.")
        except Exception as e:
            dashboard_logger.error(f"Error fetching trials from {ORACLE_API}/list_trials: {e}", exc_info=True)
            return HTMLResponse(html_template("Trials Error", f"<div class='error-message'><h3>Error fetching trials:</h3><p>Could not retrieve trial data from Oracle server. Details: {html.escape(str(e))}</p><a href='/'>← Back to Logs</a></div>"), status_code=502)

    # Display an error message if Oracle API is not reachable
    if not oracle_api_reachable:
        error_body = f"<div class='error-message'><h3>Oracle Server Unreachable</h3><p>Could not connect to the Oracle API at <strong>{ORACLE_API}</strong>. Please ensure the Oracle server (`oracle_server_main.py`) is running and accessible.</p><a href='/'>← Back to Logs</a></div>"
        return HTMLResponse(html_template("Oracle Unreachable", error_body), status_code=503)

    if not trials:
        return HTMLResponse(html_template("No Trials", "<h3>No trials available yet.</h3><p>The Oracle server is running, but no trials have been recorded or requested.</p><a href='/'>← Back to Logs</a>"))

    # Sort trials: Completed trials with scores first (sorted by score), then running, then failed, then others.
    def sort_key(t):
        status = t.get("status")
        score = t.get("score")
        if status == "COMPLETED" and score is not None:
            return (0, score) # Completed trials, sort by score (assuming lower is better)
        elif status == "RUNNING":
            return (1, 0) # Running trials, higher priority than failed
        elif status == "FAILED":
            return (2, 0) # Failed trials
        else:
            return (3, 0) # Any other status

    trials.sort(key=sort_key)

    # Determine best score only from completed trials
    best_score_trial = None
    for t in trials:
        if t.get("status") == "COMPLETED" and t.get("score") is not None:
            # Assuming lower score is better (e.g., loss)
            if best_score_trial is None or t["score"] < best_score_trial["score"]: 
                best_score_trial = t
    best_id = best_score_trial['trial_id'] if best_score_trial else None


    total_trials_count = len(trials)
    completed_trials_count = sum(t["status"] == "COMPLETED" for t in trials)
    failed_trials_count = sum(t["status"] == "FAILED" for t in trials)
    running_trials_count = sum(t["status"] == "RUNNING" for t in trials)
    
    max_configured_trials = oracle_status.get("max_trials", "N/A")
    objective_name = oracle_status.get("objective", "N/A")

    # Calculate progress percentage for the progress bar
    progress_percent = 0
    if isinstance(max_configured_trials, int) and max_configured_trials > 0:
        progress_percent = (completed_trials_count + failed_trials_count) / max_configured_trials * 100
        progress_percent = min(progress_percent, 100) # Cap at 100%
        progress_text = f"{progress_percent:.1f}% ({completed_trials_count + failed_trials_count} / {max_configured_trials})"
    else:
        progress_text = "Progress N/A (Max trials not set)"

    summary_html = f"""
    <h3>Tuning Summary (Objective: {objective_name})</h3>
    <div class="progress-bar-container">
        <div class="progress-bar" style="width: {progress_percent}%;">
            {progress_text}
        </div>
    </div>
    <ul>
        <li>Total Trials Recorded: {total_trials_count}</li>
        <li>Max Configured Trials: {max_configured_trials}</li>
        <li>Running: {running_trials_count}</li>
        <li>Completed: {completed_trials_count}</li>
        <li>Failed: {failed_trials_count}</li>
        <li>Best Score: {best_score_trial.get("score") if best_score_trial else "N/A"}</li>
    </ul>
    """

    rows = ""
    for idx, trial in enumerate(trials):
        hp_id = f"hp_{idx}"
        status_class = f"status-{trial['status']}"
        score = trial.get("score")
        # Display score with 4 decimal places if it's a number, otherwise "N/A"
        score_display = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        
        # Highlight best trial only if it's completed and is indeed the best
        row_class = "class='best-trial'" if trial["trial_id"] == best_id and trial["status"] == "COMPLETED" else ""
        
        hp_dict = trial.get('hyperparameters', {})
        # Escape hyperparameters for safe display in HTML
        hp_pretty = html.escape("\n".join(f"{safe_str(k)}: {safe_str(v)}" for k, v in hp_dict.items()), quote=True)
        rows += f"""<tr {row_class}>
            <td>{trial['trial_id']}</td>
            <td class='{status_class}'>{trial['status']}</td>
            <td>{score_display}</td>
            <td>
                <span class='toggle-btn' onclick=\"toggle('{hp_id}')\">Show/Hide</span>
                <button onclick=\"copyToClipboard('{hp_id}')\">\ud83d\udccb</button>
                <div id='{hp_id}' style='display:none; white-space:pre-wrap; font-size:smaller'>{hp_pretty}</div>
            </td>
        </tr>"""

    table = f"""
    <h2>Oracle Trial Status</h2>
    {summary_html}
    <label for="statusFilter">Filter by status:</label>
    <select id="statusFilter" onchange="filterTable()">
        <option value="">All</option>
        <option value="RUNNING">RUNNING</option>
        <option value="COMPLETED">COMPLETED</option>
        <option value="FAILED">FAILED</option>
        <option value="STOPPED">STOPPED</option>
        <option value="IDLE">IDLE</option>
    </select>
    <table id="trialTable">
        <thead>
            <tr><th>Trial ID</th><th>Status</th><th>Score</th><th>Hyperparameters</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <a href="/">\u2190 Back to Logs</a> | <a href="/api/trials">🔗 Raw JSON</a>
    """
    safe_table = table.encode('utf-8', 'replace').decode('utf-8')
    return HTMLResponse(html_template("Trial Dashboard", safe_table))

@app.get("/api/logs", response_class=JSONResponse)
def get_logs_json():
    """API endpoint to get raw log data."""
    if not LOG_FILE or not Path(LOG_FILE).exists(): # Use Path.exists() for robustness
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)
    with open(LOG_FILE, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()[-300:]
    return {"log": lines}

@app.get("/api/trials", response_class=JSONResponse)
def get_trials_json(status: str = None, skip: int = 0, limit: int = 50):
    """API endpoint to get raw trial data, with optional filtering and pagination."""
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        trials = response.json().get("trials", [])
        if status:
            trials = [t for t in trials if t["status"] == status]
        return {"trials": trials[skip:skip+limit]}
    except Exception as e:
        dashboard_logger.error(f"Error in /api/trials: {e}", exc_info=True) # Use dashboard_logger
        return JSONResponse(content={"error": str(e)}, status_code=502)

@app.get("/health", response_class=JSONResponse)
def oracle_health_check():
    """Health check endpoint for the dashboard itself, and checks Oracle status."""
    try:
        # Pings the Oracle API to check its health
        r = requests.get(f"{ORACLE_API}/status", timeout=5)
        r.raise_for_status()
        return {"status": "dashboard_alive", "oracle_status": r.json()}
    except Exception as e:
        dashboard_logger.error(f"Health check failed to reach Oracle: {e}", exc_info=True) # Use dashboard_logger
        return {"status": "dashboard_alive", "oracle_status": "error", "oracle_detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application
    uvicorn.run(app, host="192.168.1.103", port=8080)