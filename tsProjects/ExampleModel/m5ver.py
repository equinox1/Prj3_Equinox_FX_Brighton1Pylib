import MetaTrader5 as mt5
print(dir(mt5))
mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M1, datetime(2020, 1, 28), 1000)