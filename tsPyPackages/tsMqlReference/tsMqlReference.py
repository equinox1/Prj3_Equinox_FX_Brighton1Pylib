import logging
from datetime import datetime
import tzlocal
import zoneinfo  # Import zoneinfo

logger = logging.getLogger(__name__)

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

        if self.basedatatime not in constants:
            raise ValueError(f"Invalid time unit for basedatatime: {self.basedatatime}")
        if self.loaded_data_type not in constants:
            raise ValueError(f"Invalid time unit for loaded_data_type: {self.loaded_data_type}")

        base_value = constants[self.basedatatime]
        loaded_value = constants[self.loaded_data_type]

        if unit not in constants:
            raise ValueError(f"Invalid time unit requested: {unit}")

        return constants[unit] / (loaded_value / base_value)

    def get_current_time(self):
        """
        Retrieve the current time-related constants.
        """
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

    def run_service(self):
        """
        Run the reference script and log the results.
        """
        time_data = self.get_current_time()

        for key, value in time_data.items():
            logging.info(f"{key}: {value}")

        # Get time values for all defined units
        for unit in self.TIME_CONSTANTS["TIMEVALUE"].keys():
            time_value = self.get_timevalue(unit)
            logging.info(f"Time value for '{unit}': {time_value}")

        return time_data