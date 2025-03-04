import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlUtilities import CUtilities

class CDataProcess:
    def __init__(self, **kwargs):
        self._logger = logging.getLogger(__name__)
        self.util_config = CUtilities()
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()
        
        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.app_params = self.params.get("app", {})

        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 2)

        self._initialize_mql()
        self._set_global_parameters()
        self.COLUMN_PARAMS = self._define_column_params()

        # FEATURE_PARAMS must be initialized here after _set_global_parameters()
        self.FEATURE_PARAMS = {
            "df_api_ticks": {
                'close_column': 'T1_Bid_Price',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled
            },
            "df_api_rates": {
                'close_column': 'R1_Close',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
            }, 
            "df_file_ticks": {
                'close_column': 'T2_Bid_Price',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
            },
            "df_file_rates": {
                'close_column': 'R2_Close',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
            }
        }

    def _initialize_mql(self):
        self.os_platform = platform_checker.get_platform()
        self.loadmql = run_platform.RunPlatform().check_mql_state()
        self._logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")
        
        if self.loadmql:
            try:
                global mt5
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    self._logger.error(f"Failed to initialize MetaTrader5. Error: {mt5.last_error()}")
            except ImportError as e:
                self._logger.error(f"Failed to import MetaTrader5: {e}")

    def _set_global_parameters(self):
        self.mp_ml_input_keyfeat = self.ml_params.get('mp_ml_input_keyfeat', 'KeyFeature')
        self.mp_ml_input_keyfeat_scaled = self.ml_params.get('mp_ml_input_keyfeat_scaled', 'KeyFeature_Scaled')
        self.lookahead_periods = self.ml_params.get('mp_lookahead_periods', 1)
        self.ma_window = self.ml_params.get('mp_ma_window', 10)
        self.hl_avg_col = self.ml_params.get('mp_hl_avg_col', 'HL_Avg')
        self.ma_col = self.ml_params.get('mp_ma_col', 'MA')
        self.returns_col = self.ml_params.get('mp_returns_col', 'Returns')
        self.shift_in = self.ml_params.get('mp_shift_in', 1)
        self.create_label = self.ml_params.get('mp_create_label', False)
        self._logger.info(f"Global parameters set: {self.mp_ml_input_keyfeat}, {self.mp_ml_input_keyfeat_scaled}, {self.lookahead_periods}, {self.ma_window}, {self.hl_avg_col}, {self.ma_col}, {self.returns_col}, {self.shift_in}, {self.create_label}")

    def _define_column_params(self):
        return {filetype: self._generate_column_config(filetype) for filetype in ["df_api_ticks", "df_api_rates", "df_file_ticks", "df_file_rates"]}

    def _generate_column_config(self, filetype):
       if filetype == "df_api_ticks":
             prefix = 'T1'
       elif filetype == "df_api_rates":
               prefix = 'R1'
       elif filetype == "df_file_ticks":
               prefix = 'T2'
       elif filetype == "df_file_rates":
               prefix = 'R2'

       return {
            'bid_column': f'{prefix}_Bid_Price',
            'ask_column': f'{prefix}_Ask_Price',
            'column_in': f'{prefix}_Bid_Price',
            'column_out1': self.mp_ml_input_keyfeat,
            'column_out2': self.mp_ml_input_keyfeat_scaled,
            'lookahead_periods': self.lookahead_periods,
            'ma_window': self.ma_window,
            'hl_avg_col': self.hl_avg_col,
            'ma_col': self.ma_col,
            'returns_col': self.returns_col,
            'shift_in': self.shift_in,
            'create_label': self.create_label
        }

    def wrangle_time(self, df, mp_filesrc, mp_unit='s', mp_drop=True):
        mappings = {
            'ticks1': {'time': 'T1_Date'},
            'rates1': {'time': 'R1_Date'},
            'ticks2': {'time': 'T2_Date'},
            'rates2': {'time': 'R2_Date'}
        }
        rename_dict = mappings.get(mp_filesrc, {})
        if df.empty:
            self._logger.warning(f"DataFrame for {mp_filesrc} is empty!")
            return df
        
        df.rename(columns={old: new for old, new in rename_dict.items() if old in df.columns}, inplace=True)
        
        time_column = rename_dict.get('time')
        if time_column and time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce', utc=True)
        
        if mp_drop:
            df.dropna(inplace=True)
        return df

    def run_wrangle_service(self, **kwargs):
        self.df = kwargs.get('df', pd.DataFrame())
        self.df_name = kwargs.get('df_name', 'df_name')
        self.mp_unit = kwargs.get('UNIT', 's')
        self.mp_drop1 = self.data_params.get('df1_mp_data_drop',False)
        self.mp_drop2 = self.data_params.get('df2_mp_data_drop',False)
        self.mp_drop3 = self.data_params.get('df3_mp_data_drop',False)
        self.mp_drop4 = self.data_params.get('df4_mp_data_drop',False)

        self._logger.info(f"Dataframe size df: {self.df.shape} for {self.df_name}")
      
        if self.df_name == 'df_api_ticks':
            self.df = self.wrangle_time(self.df, 'ticks1',self.mp_unit,self.mp_drop1)
            self.df = self.run_average_columns(self.df, self.df_name)
        elif self.df_name == 'df_api_rates':
            self.df = self.wrangle_time(self.df, 'rates1',self.mp_unit,self.mp_drop2)
            self.df = self.run_average_columns(self.df, self.df_name)
        elif self.df_name == 'df_file_ticks':
            self.df = self.wrangle_time(self.df, 'ticks2',self.mp_unit,self.mp_drop3)
            self.df = self.run_average_columns(self.df, self.df_name)
        elif self.df_name == 'df_file_rates':
            self.df = self.wrangle_time(self.df, 'rates2',self.mp_unit,self.mp_drop4)
            self.df = self.run_average_columns(self.df, self.df_name)
        return self.df


    def run_average_columns(self, df, df_name):
        self._logger.info(f"Run Average Columns: {df_name}")
        self._logger.info(f"DataFrame columns: {df.columns}")
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        self._logger.info(f"Processing Run Average DataFrame: {df_name} with parameters: {df_params}")
        column_in = df_params.get("column_in")
        ma_col = df_params.get("ma_col")
        returns_col = df_params.get("returns_col")

        if df.empty or column_in not in df.columns:
            self._logger.warning(f"Skipping {df_name} processing. DataFrame is empty or missing {column_in}.")
            return df
        
        df[ma_col] = df[column_in].rolling(window=min(self.ma_window, len(df)), min_periods=1).mean()
        df[returns_col] = np.where(df[column_in] > 0, np.log(df[column_in] / df[column_in].shift(self.shift_in)), np.nan)
        df.dropna(subset=[returns_col], inplace=True)
        
        return df

    def establish_common_feat_col(self, df, df_name):
        df_params = self.FEATURE_PARAMS.get(df_name, {})
        self._logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        self._logger.info(f"DataFrame columns: {df.columns}")
       
        column_out1 = df_params.get("column_out1")
        self._logger.info(f"establish_common_feat_col : Column Out1: {column_out1}")
        if column_out1 and column_out1 in df.columns:
            df[column_out1] = df[column_out1]
            self._logger.info(f"Created common feature column: {column_out1}")
        return df

    def establish_common_feat_col_scaled(self, df, df_name):
        df_params = self.FEATURE_PARAMS.get(df_name, {})
        self._logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        
        column_out1 = df_params.get("column_out1")
        column_out2 = df_params.get("column_out2")
        logger.info(f"Establish_common_feat_col_scaled:Column Out1: {column_out1}, Column Out2: {column_out2}")

        if column_out1 in df.columns:
            df[column_out2] = df[column_out1].pct_change().fillna(0)
            self._logger.info(f"Created scaled column: {column_out2}")
        return df
