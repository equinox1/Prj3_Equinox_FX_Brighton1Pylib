from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import requests

app = FastAPI()

# Configuration
LOG_FILE = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir/tsneuropredict_app.log"
ORACLE_API = "http://192.168.1.103:9000"

@app.get("/", response_class=HTMLResponse)
def read_logs():
    if not os.path.exists(LOG_FILE):
        return "<h3>No log file found.</h3>"
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
    try:
        response = requests.get(f"{ORACLE_API}/list_trials", timeout=5)
        response.raise_for_status()
        trials = response.json().get("trials", [])
    except Exception as e:
        return f"<h3>Error fetching trials from Oracle server: {e}</h3>"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.103", port=8080)
