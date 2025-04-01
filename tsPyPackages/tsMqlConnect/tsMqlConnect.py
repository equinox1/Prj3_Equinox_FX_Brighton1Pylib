import logging
import keyring as kr
import os
from pathlib import Path
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize platform
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"TsMqlConnect: Running on: {os_platform}, loadmql state: {loadmql}")

if loadmql:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        mt5 = None
        loadmql = False
        logger.error("MetaTrader5 module not found. Ensure it is installed.")
else:
    mt5 = None

class CMqlBrokerConfig:
    DEFAULT_BROKERS = {
        "ICM": {
            "broker": "xerces_icm",
            "broker_path": "Brokers/ICMarkets/terminal64.exe",
            "files_path": "Brokers/ICMarkets/MQL5/Files/",
            "server": "ICMarketsSC-Demo",
            "timeout": 60000,
            "portable": True,
        },
        "METAQUOTES": {
            "broker": "xerces_meta",
            "broker_path": "Brokers/Metaquotes/terminal64.exe",
            "files_path": "Brokers/Metaquotes/MQL5/Files/",
            "server": "MetaQuotes-Demo",
            "timeout": 60000,
            "portable": True,
        },
    }

    def __init__(self, lpbroker='METAQUOTES'):
        self.lpbroker = lpbroker.upper()
        self.mt5 = mt5 if loadmql else None
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()

        # Extract base parameters
        base_params = self.params.get("base", {})
        self.broker = self.DEFAULT_BROKERS.get(self.lpbroker, {})

        self.base_path = base_params.get('mp_glob_base_connect_path', '')
        self.data_path = base_params.get('mp_glob_bae_mql_data_path', '')

        # Build and verify the full broker executable path
        base_path_obj = Path(self.base_path).resolve()
        self.broker_path = (base_path_obj / self.broker.get('broker_path', '')).resolve()
        if not self.broker_path.exists():
            error_msg = f"Broker executable not found at: {self.broker_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.file_path = self.broker.get('files_path', '')
        logger.info(f"Base Path: {base_path_obj}")
        logger.info(f"Data Path: {self.data_path}")
        logger.info(f"Broker Path: {self.broker_path}")

        self.server = self.broker.get('server', '')
        self.timeout = self.broker.get('timeout', 60000)
        self.portable = self.broker.get('portable', True)

        # Get credentials
        self.mqllogin, self.mqlpass = self.get_credentials()
        if self.mqllogin is None or self.mqlpass is None:
            raise ValueError("Credentials not found for broker.")
        logger.info(f"Broker configuration loaded: {self.broker}")

    def get_credentials(self):
        # Retrieve credentials from keyring using the broker name
        cred = kr.get_credential(self.broker.get('broker', ''), "")
        if cred:
            try:
                return int(cred.username), str(cred.password)
            except ValueError as ve:
                logger.error("Error parsing credentials: %s", ve)
                return None, None
        logger.error("Credentials not found for broker.")
        return None, None

    def run_mql_login(self):
        if not loadmql or self.mt5 is None:
            raise RuntimeError("MT5 is not available on this OS or module failed to load.")
        try:
            initialized = self.mt5.initialize(
                path=str(self.broker_path),
                login=self.mqllogin,
                password=self.mqlpass,
                server=self.server,
                timeout=self.timeout,
                portable=self.portable
            )
            if initialized:
                logger.info(f"MT5 platform launched successfully: {self.mt5.version()}")
                logger.info(f"Connected to server: {self.server}")
                logger.info(f"Terminal Info: {self.mt5.terminal_info()}")
                return True
            else:
                error_code = self.mt5.last_error()
                logger.error(f"Initialization failed: {error_code}")
                self.mt5.shutdown()
                return error_code
        except Exception as e:
            logger.exception("An exception occurred during MT5 initialization.")
            if self.mt5:
                self.mt5.shutdown()
            return str(e)

if __name__ == "__main__":
    try:
        broker_config = CMqlBrokerConfig("METAQUOTES")
        result = broker_config.run_mql_login()
        if result is True:
            print("Successfully logged in to MetaTrader 5.")
        else:
            print(f"Failed to login. Error code: {result}")
    except Exception as e:
        print(f"Error: {e}")
        