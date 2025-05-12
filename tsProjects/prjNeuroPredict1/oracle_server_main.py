# ✅ oracle_server_main.py
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
import logging
import time
import threading
import uvicorn

logger = logging.getLogger("OracleServer")
logging.basicConfig(level=logging.INFO)

def run_uvicorn(app, host, port):
    uvicorn.run(app, host=host, port=port, log_level="info")

def main():
    try:
        logger.info("\U0001f9e0 Creating CustomOracle...")
        oracle = CustomOracle(objective="val_loss", max_trials=50)

        logger.info("\U0001f680 Starting OracleServer at http://192.168.1.103:9000 ...")
        server = OracleServer(oracle, tuner_id="chief")

        thread = threading.Thread(target=run_uvicorn, args=(server.app, "192.168.1.103", 9000), daemon=True)
        thread.start()

        logger.info("✅ OracleServer is now running.")
        while True:
            time.sleep(60)

    except Exception as e:
        logger.error(f"❌ Failed to start OracleServer: {e}")
        logger.info("⚙️ Cleaning up resources...")
        logger.info("✅ Cleanup completed.")

if __name__ == "__main__":
    main()
