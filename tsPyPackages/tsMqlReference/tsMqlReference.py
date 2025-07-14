import logging
import os
from datetime import datetime
import tzlocal
import zoneinfo  # Import zoneinfo

# -- start of logging setup --
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

logger = logging.getLogger(__name__)

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

global_logdir = app_params.get('mp_glob_base_log_path', './Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

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

        return (constants[unit] * base_value) / loaded_value

    def mt5_timeframe_from_string(self, timeframe_str):
        """
        Converts a string representation of a timeframe to its MetaTrader5 equivalent.
        :param timeframe_str: The string representation of the timeframe (e.g., 'M1', 'H4').
        :return: The MetaTrader5 timeframe constant (e.g., mt5.TIMEFRAME_M1) as an integer.
                 Raises ValueError if MetaTrader5 is not available or if the timeframe string is not recognized.
        """
        if not mt5:
            logger.critical("MetaTrader5 is not initialized. Cannot convert timeframe string to MT5 constant.")
            raise ValueError("MetaTrader5 is not initialized. Please ensure mt5 is installed and initialized.")

        # Directly map string to MetaTrader5 integer constants
        timeframe_mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        mt5_timeframe = timeframe_mapping.get(timeframe_str)

        if mt5_timeframe is None:
            logger.error(f"Unsupported timeframe string '{timeframe_str}'. Valid options are: {list(timeframe_mapping.keys())}")
            raise ValueError(f"Unsupported timeframe string '{timeframe_str}' for MetaTrader5.")

        return mt5_timeframe


    def get_current_time(self):
        """
        Retrieve the current time-related constants.
        Note: The 'TIMEFRAME' key here is for informational purposes (e.g., logging)
        and returns the string representation, not the MT5 integer constant.
        For MT5 constant, use mt5_timeframe_from_string.
        """
        try:
            # We retain the string representation for 'TIMEFRAME' in this dictionary
            # as it might be used for display/info, not directly for MT5 API calls.
            # If the actual MT5 constant is needed, mt5_timeframe_from_string should be called.
            timeframe_display_str = self.required_data_type # Simply use the input required_data_type string

            return {
                "MINUTE": int(self.get_timevalue('MINUTE')),
                "HOUR": int(self.get_timevalue('HOUR')),
                "DAY": int(self.get_timevalue('DAY')),
                "TIMEZONE": self.local_timezone,
                "TIMEFRAME": timeframe_display_str, # This remains a string for display/info
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