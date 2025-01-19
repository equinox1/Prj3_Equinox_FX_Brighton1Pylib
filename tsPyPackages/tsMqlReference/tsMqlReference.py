#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

# classes for mql

import MetaTrader5 as mt5

class CMqlTimeConfig:
    TIME_CONSTANTS = {
        "TIMEVALUE": {
            'SECONDS': 1,
            'MINUTES': 60,
            'HOURS': 60 * 60,
            'DAYS': 24 * 60 * 60,
            'WEEKS': 7 * 24 * 60 * 60,
            'YEARS': 365.25 * 24 * 60 * 60  # Average year length
        },
        "TIMEFRAME": {
            'M1': "mt5.TIMEFRAME_M1",
            'M5': "mt5.TIMEFRAME_M5",
            'M15': "mt5.TIMEFRAME_M15",
            'M30': "mt5.TIMEFRAME_M30",
            'H1': "mt5.TIMEFRAME_H1",
            'H4': "mt5.TIMEFRAME_H4",
            'D1': "mt5.TIMEFRAME_D1",
            'W1': "mt5.TIMEFRAME_W1",
            'MN1': "mt5.TIMEFRAME_MN1"
        },
        "UNIT": [
            ('Milliseconds', 'ms'), ('Seconds', 's'), ('Minutes', 'm'), 
            ('Hours', 'h'), ('Days', 'd'), ('Weeks', 'w'), ('Months', 'm')
        ],
        "DATATYPE": {
            'TICKS': 'TICKS', 'MINUTES': 'MINUTES', 'HOURS': 'HOURS',
            'DAYS': 'DAYS', 'WEEKS': 'WEEKS', 'MONTHS': 'MONTHS', 'YEARS': 'YEARS'
        },
        "SYMBOLS": [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
            "EURJPY", "EURGBP", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
            "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
            "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
            "NZDJPY", "NZDCAD", "NZDCHF",
            "CADJPY", "CADCHF", "CHFJPY"
        ],
        "TIMEZONES": [
            "etc/UTC", "Europe/London", "America/New_York", "America/Chicago",
            "America/Denver", "America/Los_Angeles", "Asia/Tokyo",
            "Asia/Hong_Kong", "Asia/Shanghai", "Asia/Singapore",
            "Asia/Dubai", "Asia/Mumbai", "Australia/Sydney",
            "Australia/Melbourne", "Africa/Johannesburg",
            "Europe/Berlin", "Europe/Paris", "Europe/Madrid",
            "Europe/Rome", "Europe/Amsterdam", "Europe/Brussels",
            "Europe/Stockholm", "Europe/Oslo", "Europe/Copenhagen",
            "Europe/Zurich", "Europe/Vienna", "Europe/Warsaw",
            "Europe/Prague", "Europe/Budapest", "Europe/Bucharest",
            "Europe/Sofia", "Europe/Athens", "Europe/Helsinki",
            "Europe/Tallinn", "Europe/Vilnius", "Europe/Riga",
            "Europe/Lisbon", "Europe/Dublin", "Europe/Edinburgh",
            "Europe/Ljubljana", "Europe/Bratislava", "Europe/Luxembourg",
            "Europe/Monaco", "Europe/Valletta", "Europe/Andorra",
            "Europe/San_Marino", "Europe/Vatican", "Europe/Gibraltar"
        ]
    }

    def __init__(self, basedatatime='SECONDS', loadeddatatime='MINUTES', **kwargs):
        """
        Initialize the CMqlTimeConfig instance.

        :param basedatatime: The base data time unit (default is 'SECONDS').
        :param loadeddatatime: The loaded data time unit (default is 'MINUTES').
        :param kwargs: Additional keyword arguments.
        """
        self.basedatatime = basedatatime
        self.loadeddatatime = loadeddatatime
        self.kwargs = kwargs

    def get_timevalue(self, unit):
        """
        Get the time value for a given unit.

        :param unit: The unit of time.
        :return: The adjusted time value based on `basedatatime` and `loadeddatatime`.
        """
        constants = self.TIME_CONSTANTS["TIMEVALUE"]

        if self.basedatatime not in constants or self.loadeddatatime not in constants:
            raise ValueError("Invalid time unit for basedatatime or loadeddatatime.")

        base_value = constants[self.basedatatime]
        loaded_value = constants[self.loadeddatatime]

        if unit not in constants:
            raise ValueError("Invalid time unit requested.")

        shift = loaded_value / base_value
        timevalue = constants[unit] / shift
        
        return timevalue

