from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import html

# Configuration
LOG_FILE = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir/tsneuropredict_app.log"
ORACLE_API = "http://192.168.1.103:9000"

app = FastAPI(title="Tuner Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def html_template(title: str, body: str) -> str:
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: monospace; padding: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .logbox {{ white-space: pre-wrap; background: #f9f9f9; border: 1px solid #ccc; padding: 10px; height: 600px; overflow-y: scroll; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f0f0f0; }}
                a {{ display: inline-block; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                {body}
            </div>
        </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def show_logs():
    if not os.path.exists(LOG_FILE):
        return HTMLResponse(html_template("Log Viewer", "<h3>No log file found.</h3>"), status_code=404)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]

    escaped_log = html.escape("".join(lines))
    body = f"<h2>Tuning Logs (Live)</h2><div class='logbox'>{escaped_log}</div><a href='/trials'>→ View Trials Dashboard</a>"
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

    rows = ""
    for trial in trials:
        hp_str = html.escape(", ".join(f"{k}={v}" for k, v in trial.get('hyperparameters', {}).items()))
        score = trial.get("score", "")
        rows += f"<tr><td>{trial['trial_id']}</td><td>{trial['status']}</td><td>{score}</td><td>{hp_str}</td></tr>"

    table = f"""
    <h2>Oracle Trial Status</h2>
    <table>
        <tr><th>Trial ID</th><th>Status</th><th>Score</th><th>Hyperparameters</th></tr>
        {rows}
    </table>
    <a href="/">← Back to Logs</a>
    """
    return HTMLResponse(html_template("Trial Dashboard", table))


@app.get("/api/logs", response_class=JSONResponse)
def get_logs_json():
    if not os.path.exists(LOG_FILE):
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]
    return {"log": lines}


@app.get("/api/trials", response_class=JSONResponse)
def get_trials_json():
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.103", port=8080)
