from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging

# Configuration
LOG_FILE = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir/tsneuropredict_app.log"
ORACLE_API = "http://192.168.1.103:9000"

# Initialize FastAPI app
app = FastAPI(title="Tuner Dashboard")

# Enable CORS (optional, helps with frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.get("/", response_class=HTMLResponse)
def read_logs():
    """Display the latest 300 lines of the log file as HTML."""
    if not os.path.exists(LOG_FILE):
        return HTMLResponse("<h3>No log file found.</h3>", status_code=404)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]

    html_lines = "<br>".join(line.replace(" ", "&nbsp;") for line in lines)

    return f"""
    <html>
        <head>
            <title>Tuner Log</title>
            <meta http-equiv="refresh" content="5">
        </head>
        <body style="font-family: monospace;">
            <h2>Tuning Log (Live)</h2>
            <div style="white-space: pre; height: 600px; overflow-y: scroll; border: 1px solid #ccc;">
                {html_lines}
            </div>
            <p><a href="/trials">→ View Trials Dashboard</a></p>
        </body>
    </html>
    """


@app.get("/trials", response_class=HTMLResponse)
def show_trials():
    """Render a table of current trials from the Oracle API."""
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        trials = response.json().get("trials", [])
    except Exception as e:
        logging.error(f"Failed to fetch trials: {e}")
        return HTMLResponse(f"<h3>Error fetching trials from Oracle server: {e}</h3>", status_code=502)

    rows = ""
    for trial in trials:
        hp_str = ", ".join(f"{k}={v}" for k, v in trial['hyperparameters'].items())
        score = trial.get("score", "")
        rows += f"<tr><td>{trial['trial_id']}</td><td>{trial['status']}</td><td>{score}</td><td>{hp_str}</td></tr>"

    return f"""
    <html>
        <head>
            <title>Trial Status Dashboard</title>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <h2>Oracle Trial Statuses</h2>
            <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Trial ID</th><th>Status</th><th>Score</th><th>Hyperparameters</th>
                </tr>
                {rows}
            </table>
            <p><a href="/">← Back to Logs</a></p>
        </body>
    </html>
    """


@app.get("/api/logs", response_class=JSONResponse)
def get_logs_json():
    """Serve the last 300 lines of the log as JSON."""
    if not os.path.exists(LOG_FILE):
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-300:]
    return {"log": lines}


@app.get("/api/trials", response_class=JSONResponse)
def get_trials_json():
    """Serve trials data from the Oracle API as JSON."""
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.103", port=8080)
