import pandas as pd
import numpy as np
import datetime
from datetime import datetime

# Ratesdata
# Date,Timestamp,Open,High,Low,Close,Volume
# 20030505,00:00:00,1.12161,1.12209,1.12161,1.12209,16,16,6

# Original data
ratesdata = {
    "Date": ["20030505"],
    "Timestamp": ["00:00:00"],
    "Open": [1.12161],
    "High": [1.12209],
    "Low": [1.12161],
    "Close": [1.12209],
    "Volume": [16],
}

# Tickdata
# Date,Timestamp,Bid Price,Ask Price,Last Price,Volume
# 20240101,22:00:12,1.10427,1.10481,1.10427,1

# Create the data
tickdata = {
    "Date": ["20240101"],
    "Timestamp": ["22:00:12"],
    "Bid Price": [1.10427],
    "Ask Price": [1.10481],
    "Last Price": [1.10427],
    "Volume": [1]
}

# Create a DataFrame
df1 = pd.DataFrame(ratesdata)
df2 = pd.DataFrame(tickdata)
fswitch = 1

if fswitch == 1:
    # Convert the Date and Timestamp columns to datetime
    df1['Date'] = pd.to_datetime(df1['Date'], format="%Y%m%d", errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], format="%Y%m%d", errors='coerce')
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)

elif fswitch == 2:
    # Convert the Date and Timestamp columns to datetime
    df1['Date'] = pd.to_datetime(df1['Date'], unit='s', errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], unit='s', errors='coerce')
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)

elif fswitch == 3:
    df1['Date'] = pd.to_datetime(df1['Date'].astype(str))
    df2['Date'] = pd.to_datetime(df2['Date'].astype(str))
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)

fswitch = 3
if fswitch == 1:
    # Convert the Date and Timestamp columns to datetime
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], format="%H:%M:%S", errors='coerce')
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], format="%H:%M:%S", errors='coerce')
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)

elif fswitch == 2:
    # Convert the Date and Timestamp columns to datetime
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], unit='s', errors='coerce')
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], unit='s', errors='coerce')
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)

elif fswitch == 3:
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'].astype(str))
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'].astype(str))
    # Display the result
    print("Ratesdata")
    print(df1)
    print("Tickdata")
    print(df2)