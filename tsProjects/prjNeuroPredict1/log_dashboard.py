
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()
LOG_FILE = r"C:/WinRunMnt1/8.0 Projects/8.3 ProjectModelsEquinox/EQUINRUN/Logdir/tsneuropredict_app.log"  # Change if dynamic

@app.get("/", response_class=HTMLResponse)
def read_logs():
    if not os.path.exists(LOG_FILE):
        return "<h3>No log file found.</h3>"
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-200:]
    html_lines = "<br>".join(line.replace(" ", "&nbsp;") for line in lines)
    return f"<html><body><h2>Live Logs</h2><div style='font-family: monospace;'>{html_lines}</div></body></html>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.103", port=8080)
