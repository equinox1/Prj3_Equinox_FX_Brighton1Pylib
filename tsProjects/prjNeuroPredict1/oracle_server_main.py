# oracle_server_main.py
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
import logging
import time

def main():
    logger = logging.getLogger("OracleServer")

    try:
        logger.info("Creating CustomOracle...")
        oracle = CustomOracle(objective="val_loss", max_trials=50)

        logger.info("Starting OracleServer at http://192.168.1.103:9000 ...")
        server = OracleServer(oracle)
        server.start(host="192.168.1.103", port=9000)
        logger.info("✅ OracleServer is now running.")
        
        # Keep the main thread alive
        while True:
            time.sleep(60)

    except Exception as e:
        logger.error(f"❌ Failed to start OracleServer: {e}")

if __name__ == "__main__":
    main()