import logging
<<<<<<< HEAD
import os
from tsMqlSetup import CMqlSetup # Correctly import the class

# --- Logging setup ---
# This script now *only* gets a logger. The root logger is configured by multiworker_launcher.py.
# This prevents repeated "Logging initialized" messages and ensures a consistent log file.
logger = logging.getLogger(__name__)
# -- end of logging setup ----
from datetime import datetime
import tzlocal
import zoneinfo  # Import zoneinfo


# -- start of logging setup --
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')
=======
from datetime import datetime
import tzlocal
import zoneinfo  # Import zoneinfo
import os

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides

env_backend = os.environ.get("MLTUNE_BACKEND", "tensorflow")
env_gtuner = os.environ.get("GTUNER_MODEL", env_backend)

mql_overrides = CMqlOverrides()
mql_overrides.env.override_params({
    "mltune": {"backend": env_backend},
    "app": {"gtuner_model": env_gtuner}
})

app_params = mql_overrides.env.all_params().get("app", {})
gtuner_model = app_params.get('gtuner_model', 'pytorch')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)

global_logdir, global_logfile = setup_config.set_log_dir(
    logdir=None,
    logfile=xerces_logfile,
    servername=xerces_servername,
    ltuner=gtuner_model
)

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2


try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Handle gracefully if MetaTrader5 is not available

class CMqlRefConfig:
    def __init__(self, loaded_data_type='MINUTE', required_data_type='H4', **kwargs):
        """
        Initialize the CMqlRefConfig instance.

        :param loaded_data_type: The loaded data time unit (default is 'MINUTE').
        :param required_data_type: The timeframe for market data (default is 'H4').
        :param kwargs: Additional keyword arguments.
        """
        self.basedatatime = kwargs.get('basedatatime', 'SECOND')
        self.loaded_data_type = loaded_data_type
        self.required_data_type = required_data_type
        local_zone = tzlocal.get_localzone()
        if isinstance(local_zone, zoneinfo.ZoneInfo):
            self.local_timezone = local_zone.key  # Access the timezone key
        else:
            self.local_timezone = local_zone  # Old behavior, if applicable

    TIME_CONSTANTS = {
        "TIMEVALUE": {
            'SECOND': 1,
            'MINUTE': 60,
            'HOUR': 3600,
            'DAY': 86400,
            'WEEK': 604800,
            'YEAR': 31557600  # Approximate average year length
        },
        "TIMEFRAME": {
            'M1': "mt5.TIMEFRAME_M1" if mt5 else "TIMEFRAME_M1",
            'M5': "mt5.TIMEFRAME_M5" if mt5 else "TIMEFRAME_M5",
            'M15': "mt5.TIMEFRAME_M15" if mt5 else "TIMEFRAME_M15",
            'M30': "mt5.TIMEFRAME_M30" if mt5 else "TIMEFRAME_M30",
            'H1': "mt5.TIMEFRAME_H1" if mt5 else "TIMEFRAME_H1",
            'H4': "mt5.TIMEFRAME_H4" if mt5 else "TIMEFRAME_H4",
            'D1': "mt5.TIMEFRAME_D1" if mt5 else "TIMEFRAME_D1",
            'W1': "mt5.TIMEFRAME_W1" if mt5 else "TIMEFRAME_W1",
            'MN1': "mt5.TIMEFRAME_MN1" if mt5 else "TIMEFRAME_MN1"
        },
        "UNIT": {
            'SECOND': 's',
            'MINUTE': 'm',
            'HOUR': 'h',
            'DAY': 'd',
            'WEEK': 'w',
            'YEAR': 'y'
        }
    }

    def get_timevalue(self, unit):
        """
        Get the time value for a given unit.

        :param unit: The unit of time.
        :return: The adjusted time value based on `basedatatime` and `loaded_data_type`.
        """
        constants = self.TIME_CONSTANTS["TIMEVALUE"]

        # Validate `basedatatime` and `loaded_data_type`
        if self.basedatatime not in constants:
            logger.error(f"Invalid time unit for basedatatime: {self.basedatatime}")
            raise ValueError(f"Invalid time unit for basedatatime: {self.basedatatime}")
        if self.loaded_data_type not in constants:
            logger.error(f"Invalid time unit for loaded_data_type: {self.loaded_data_type}")
            raise ValueError(f"Invalid time unit for loaded_data_type: {self.loaded_data_type}")

        base_value = constants[self.basedatatime]
        loaded_value = constants[self.loaded_data_type]

        # Validate the requested unit
        if unit not in constants:
            logger.error(f"Invalid time unit requested: {unit}")
            raise ValueError(f"Invalid time unit requested: {unit}")

        return constants[unit] / (loaded_value / base_value)

    def get_current_time(self):
        """
        Retrieve the current time-related constants.
        """
        try:
            return {
                "MINUTE": int(self.get_timevalue('MINUTE')),
                "HOUR": int(self.get_timevalue('HOUR')),
                "DAY": int(self.get_timevalue('DAY')),
                "TIMEZONE": self.local_timezone,
                "TIMEFRAME": self.TIME_CONSTANTS['TIMEFRAME'].get(self.required_data_type, "TIMEFRAME_H4"),
                "CURRENTYEAR": datetime.now().year,
                "CURRENTDAY": datetime.now().day,
                "CURRENTMONTH": datetime.now().month
            }
        except Exception as e:
            logger.error(f"Error retrieving current time: {e}")
            raise

    def run_service(self):
        """
        Run the reference script and log the results.
        """
        try:
            time_data = self.get_current_time()

            for key, value in time_data.items():
                logger.info(f"{key}: {value}")

            # Get time values for all defined units
            for unit in self.TIME_CONSTANTS["TIMEVALUE"].keys():
                time_value = self.get_timevalue(unit)
                logger.info(f"Time value for '{unit}': {time_value}")

            return time_data
        except Exception as e:
            logger.error(f"Error running service: {e}")
            raise