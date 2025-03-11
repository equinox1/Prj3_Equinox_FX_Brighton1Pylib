import logging
import keyring as kr
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
import pathlib
import os

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
        
        # Extract parameters
        base_params = self.params.get("base", {})
        self.broker = self.DEFAULT_BROKERS.get(self.lpbroker, {})
        
        self.base_path = base_params.get('mp_glob_mql_basepath', '')
        self.data_path = base_params.get('mp_glob_mql_data_path', '')
        self.broker_path = os.path.join(self.base_path, self.broker.get('broker_path', ''))
        self.file_path = self.broker.get('files_path', '')
        logger.info(f"Base Path: {self.base_path}")
        logger.info(f"Data Path: {self.data_path}")
        logger.info(f"Broker Path: {self.broker_path}")

        self.server = self.broker.get('server', '')
        self.timeout = self.broker.get('timeout', 60000)
        self.portable = self.broker.get('portable', True)
        
        self.mqllogin, self.mqlpass = self.get_credentials()
        
        logger.info(f"Broker: {self.broker}, Path: {self.broker_path}")
        
    def get_credentials(self):
        cred = kr.get_credential(self.broker.get('broker', ''), "")
        if cred:
            return int(cred.username), str(cred.password)
        logger.error("Credentials not found for broker.")
        return None, None
    
    def run_mql_login(self):
        if not loadmql:
            raise RuntimeError("CMqlinit: MT5 is not available on this OS.")

        if self.mt5.initialize(
            path=self.broker_path,
            login=self.mqllogin,
            password=self.mqlpass,
            server=self.server,
            timeout=self.timeout,
            portable=self.portable
        ):
            logger.info(f"MT5 platform launched successfully: {self.mt5.version()}")
            logger.info(f"Connected to server: {self.server}")
            logger.info(f"Terminal Info: {self.mt5.terminal_info()}")
            return True
        else:
            error_code = self.mt5.last_error()
            logger.error(f"Initialization failed: {error_code}")
            self.mt5.shutdown()
            return error_code

if __name__ == "__main__":
    broker_config = CMqlBrokerConfig("METAQUOTES")
    result = broker_config.run_mql_login()
    if result is True:
        print("Successfully logged in to MetaTrader 5.")
    else:
        print(f"Failed to login. Error code: {result}")
