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

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Dynamically resolve log file
def resolve_logfile():
    base_path = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir"
    matches = glob.glob(os.path.join(base_path, "**", "tsneuropredict_app.log"), recursive=True)
    if matches:
        return max(matches, key=os.path.getmtime)
    return None

LOG_FILE = resolve_logfile()
ORACLE_API = os.getenv("ORACLE_API", "http://192.168.1.103:9000")

app = FastAPI(title="Tuner Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def html_template(title: str, body: str, extra_scripts: str = "") -> str:
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <meta id='refresh-meta' http-equiv="refresh" content="10">
            <script>
            function toggle(id) {{
                const el = document.getElementById(id);
                el.style.display = el.style.display === 'none' ? 'block' : 'none';
            }}
            function filterTable() {{
                const filter = document.getElementById('statusFilter').value;
                const rows = document.querySelectorAll("table tr");
                rows.forEach((row, index) => {{
                    if (index === 0) return;
                    const statusCell = row.cells[1];
                    const show = !filter || statusCell.textContent.trim() === filter;
                    row.style.display = show ? "" : "none";
                }});
            }}
            function copyToClipboard(id) {{
                const el = document.getElementById(id);
                navigator.clipboard.writeText(el.innerText);
            }}
            function toggleRefresh() {{
                const meta = document.getElementById('refresh-meta');
                meta.content = document.getElementById('autorefresh').checked ? "10" : "";
            }}
            {extra_scripts}
            </script>
            <style>
                body {{ font-family: monospace; padding: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .logbox {{ white-space: pre-wrap; background: #f9f9f9; border: 1px solid #ccc; padding: 10px; height: 600px; overflow-y: scroll; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f0f0f0; }}
                a {{ display: inline-block; margin-top: 15px; }}
                .status-RUNNING {{ color: orange; font-weight: bold; }}
                .status-COMPLETED {{ color: green; font-weight: bold; }}
                .status-FAILED {{ color: red; font-weight: bold; }}
                .toggle-btn {{ cursor: pointer; color: blue; text-decoration: underline; }}
                .best-trial {{ background-color: #dff0d8 !important; }}
            </style>
        </head>
        <body>
            <div class="container">
                <label><input type="checkbox" id="autorefresh" checked onchange="toggleRefresh()"> Auto-refresh</label>
                {body}
            </div>
        </body>
    </html>
    """

def safe_str(obj):
    try:
        return str(obj)
    except Exception:
        return repr(obj)

@app.get("/", response_class=HTMLResponse)
def show_logs():
    if not LOG_FILE or not os.path.exists(LOG_FILE):
        return HTMLResponse(html_template("Log Viewer", "<h3>No log file found.</h3>"), status_code=404)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]

    escaped_log = html.escape("".join(lines))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = f"<h2>Tuning Logs (Live)</h2><p><em>Last updated: {timestamp}</em></p><div class='logbox'>{escaped_log}</div><a href='/trials'>→ View Trials Dashboard</a>"
    return HTMLResponse(html_template("Log Viewer", body))

@app.get("/trials", response_class=HTMLResponse)
def show_trials():
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        trials = response.json().get("trials", [])
    except Exception as e:
        return HTMLResponse(html_template("Trials Error", f"<h3>Error: {html.escape(str(e))}</h3>"), status_code=502)

    if not trials:
        return HTMLResponse(html_template("No Trials", "<h3>No trials available yet.</h3><a href='/'>← Back to Logs</a>"))

    trials.sort(key=lambda t: t.get('score') if t.get('score') is not None else float('inf'))
    best_id = trials[0]['trial_id'] if trials[0].get("score") is not None else None

    summary = {
        "total": len(trials),
        "completed": sum(t["status"] == "COMPLETED" for t in trials),
        "failed": sum(t["status"] == "FAILED" for t in trials),
        "best_score": min((t.get("score", float("inf")) for t in trials if t.get("score") is not None), default="N/A")
    }

    summary_html = f"""
    <h3>Summary</h3>
    <ul>
        <li>Total Trials: {summary['total']}</li>
        <li>Completed: {summary['completed']}</li>
        <li>Failed: {summary['failed']}</li>
        <li>Best Score: {summary['best_score']}</li>
    </ul>
    """

    rows = ""
    for idx, trial in enumerate(trials):
        hp_id = f"hp_{idx}"
        status_class = f"status-{trial['status']}"
        score = trial.get("score", "")
        score_style = "color:green;font-weight:bold" if isinstance(score, (int, float)) and score < 0.05 else ""
        row_class = "class='best-trial'" if trial["trial_id"] == best_id else ""
        hp_dict = trial.get('hyperparameters', {})
        hp_pretty = html.escape("\n".join(f"{safe_str(k)}: {safe_str(v)}" for k, v in hp_dict.items()), quote=True)
        rows += f"""<tr {row_class}>
            <td>{trial['trial_id']}</td>
            <td class='{status_class}'>{trial['status']}</td>
            <td style='{score_style}'>{score}</td>
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
    </select>
    <table id="trialTable">
        <tr><th>Trial ID</th><th>Status</th><th>Score</th><th>Hyperparameters</th></tr>
        {rows}
    </table>
    <a href="/">\u2190 Back to Logs</a> | <a href="/api/trials">🔗 Raw JSON</a>
    """
    safe_table = table.encode('utf-8', 'replace').decode('utf-8')
    return HTMLResponse(html_template("Trial Dashboard", safe_table))

@app.get("/api/logs", response_class=JSONResponse)
def get_logs_json():
    if not LOG_FILE or not os.path.exists(LOG_FILE):
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]
    return {"log": lines}

@app.get("/api/trials", response_class=JSONResponse)
def get_trials_json(status: str = None, skip: int = 0, limit: int = 50):
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        trials = response.json().get("trials", [])
        if status:
            trials = [t for t in trials if t["status"] == status]
        return {"trials": trials[skip:skip+limit]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=502)

@app.get("/health", response_class=JSONResponse)
def oracle_health():
    try:
        r = requests.get(f"{ORACLE_API}/heartbeat", timeout=5)
        return {"status": "alive" if r.status_code == 200 else "unresponsive"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.103", port=8080)