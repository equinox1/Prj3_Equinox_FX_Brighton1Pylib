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
    def __init__(self):
        # Initialize instance variables or configuration
        pass

    @staticmethod
    def get_constants():
        # Constant Definitions
        TIMEVALUE = {
            'SECONDS': 1,
            'MINUTES': 60,
            'HOURS': 60 * 60,
            'DAYS': 24 * 60 * 60,
            'WEEKS': 7 * 24 * 60 * 60,
            'YEARS': 365.25 * 24 * 60 * 60  # Average year length accounting for leap years
        }

        TIMEFRAME = [
            mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, 
            mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, 
            mt5.TIMEFRAME_D1, mt5.TIMEFRAME_W1, mt5.TIMEFRAME_MN1
        ]

        UNIT = ['s', 'm', 'h', 'd', 'w', 'm']
        DATATYPE = ['TICKS', 'MINUTES', 'HOURS', 'DAYS', 'WEEKS', 'MONTHS']
        SYMBOLS = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", 
            "EURJPY", "EURGBP", "EURCHF", "EURCAD", "EURAUD", "EURNZD", 
            "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF", 
            "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF", 
            "NZDJPY", "NZDCAD", "NZDCHF", 
            "CADJPY", "CADCHF", "CHFJPY"
        ]

        TIMEZONES = [
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

        return {
            "TIMEVALUE": TIMEVALUE,
            "TIMEFRAME": TIMEFRAME,
            "UNIT": UNIT,
            "DATATYPE": DATATYPE,
            "SYMBOLS": SYMBOLS,
            "TIMEZONES": TIMEZONES
        }
