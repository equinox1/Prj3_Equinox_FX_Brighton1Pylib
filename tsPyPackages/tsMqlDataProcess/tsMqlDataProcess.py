import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import textwrap

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CDataProcess:
    def __init__(self, debug,datenv, globalenv, **kwargs):
        self.kwargs = kwargs
        self.debug = debug
        self.dataenv = datenv
        self.globalenv = globalenv

        if self.debug:
           print("tunerparams ",kwargs)
    
    @staticmethod
    def ensure_column_exists(df, column_name):
        """Checks if a column exists in the DataFrame."""
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns}")

    def wrangle_time(self, df: pd.DataFrame, mp_filesrc: str) -> pd.DataFrame:
        """
        Cleans and processes time-related data.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a Pandas DataFrame.")
        
        logger.info(f"Processing time data for {mp_filesrc}")
        
        time_mappings = {
            'ticks1': {'time': 'T1_Date', 'bid': 'T1_Bid_Price', 'ask': 'T1_Ask_Price'},
            'rates1': {'time': 'R1_Date', 'open': 'R1_Open', 'close': 'R1_Close'}
        }
        
        if mp_filesrc in time_mappings:
            df.rename(columns={k: v for k, v in time_mappings[mp_filesrc].items() if k in df.columns}, inplace=True)
        
        # Convert datetime columns
        if 'T1_Date' in df.columns:
            df['T1_Date'] = pd.to_datetime(df['T1_Date'], errors='coerce')
        if 'R1_Date' in df.columns:
            df['R1_Date'] = pd.to_datetime(df['R1_Date'], errors='coerce')
        
        df.dropna(inplace=True)
        return df

    def calculate_moving_average(self, df, column, window):
        """Computes a moving average."""
        self.ensure_column_exists(df, column)
        df['SMA'] = df[column].rolling(window=window, min_periods=1).mean().fillna(method='bfill')
        return df['SMA']
    
    def calculate_log_returns(self, df, column, shift):
        """Computes log returns while ensuring non-zero values."""
        self.ensure_column_exists(df, column)
        df[column] = df[column].fillna(method='ffill')
        if (df[column] <= 0).any():
            raise ValueError("Log returns require all values to be positive.")
        return np.log(df[column] / df[column].shift(shift)).dropna()

    def create_label(self, df, bid_col='Bid', ask_col='Ask', label_col='Label', shift_period=1):
        """Creates a label column using mid-price calculation."""
        self.ensure_column_exists(df, bid_col)
        self.ensure_column_exists(df, ask_col)
        df[label_col] = (df[bid_col] + df[ask_col]) / 2
        df[label_col] = df[label_col].shift(-shift_period)
        df.dropna(inplace=True)
        return df

    def normalize_data(self, df, columns_to_scale):
        """Normalizes selected columns using MinMaxScaler."""
        scaler = MinMaxScaler()
        self.ensure_column_exists(df, columns_to_scale[0])  # Ensure first column exists
        df[[f'{col}_Scaled' for col in columns_to_scale]] = scaler.fit_transform(df[columns_to_scale])
        return df, scaler

    def run_mql_show(self, df, num_rows=5, col_width=20, tablefmt='pretty'):
        """Displays a tabulated preview of the DataFrame."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.info("DataFrame is empty or invalid.")
            return
        
        def wrap_text(column, width):
            return column.apply(lambda x: '\n'.join(textwrap.wrap(str(x), width)) if pd.notnull(x) else "")
        
        df = df.apply(lambda col: wrap_text(col, col_width))
        logger.info(tabulate(df.head(num_rows), headers='keys', tablefmt=tablefmt))

    def split_data(self, X, y, val_split=0.2, test_split=0.1):
        """Splits data into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_split + test_split), shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_split / (val_split + test_split)), shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test
