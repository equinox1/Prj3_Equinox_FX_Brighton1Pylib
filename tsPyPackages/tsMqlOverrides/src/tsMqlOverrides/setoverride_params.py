import sys
import platform
import os
from pathlib import Path
import yaml  # For loading configurations

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLTunerParams import CMqlEnvMLTunerParams
from tsMqlDataParams import CMqlEnvDataParams
from tsMqlEnvCore import CEnvCore

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")

class CMqlOverrides(CEnvCore):
    """
    Environment setup for MQL-based platform, handling paths and configurations.
    """

    def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization

        self.mt5 = None
        if loadmql:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
                if not self.mt5.initialize():
                    logger.error("Failed to initialize MetaTrader5: %s", self.mt5.last_error())
            except ImportError:
                logger.error("MetaTrader5 module not found. Exiting...")
                sys.exit(1)

        self._initialize_env_mgr()
        self._set_data_overrides()
        self._set_ml_overrides()
        self._set_mltune_overrides()

    def _initialize_env_mgr(self):
        """
        Initializes the environment manager and loads parameters.
        """
        try:
            self.env = CMqlEnvMgr()  # Initialize environment manager
            self.params = self.env.all_params()
        except Exception as e:
            logger.critical("Failed to initialize CMqlEnvMgr: %s", e)
            self.params = {}

        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

        logger.info("Environment Manager initialized successfully.")

    def _set_data_overrides(self):
        """
        Overrides default data parameters for better performance and control.
        """
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping data overrides.")
            return

        data_overrides = {
            "mp_data_timeframe": "M1",
            "mp_data_rows": 1000,
            "mp_data_rowcount": 40000,
            # API Ticks Data Filters
            "df1_mp_data_filter_int": False,
            "df1_mp_data_filter_flt": False,
            "df1_mp_data_filter_obj": False,
            "df1_mp_data_filter_dtmi": False,
            "df1_mp_data_filter_dtmf": False,
            "df1_mp_data_dropna": False,
            "df1_mp_data_merge": False,
            "df1_mp_data_convert": False,
            "df1_mp_data_drop": False,
            # API Rates Data Filters
            "df2_mp_data_filter_int": False,
            "df2_mp_data_filter_flt": False,
            "df2_mp_data_filter_obj": False,
            "df2_mp_data_filter_dtmi": False,
            "df2_mp_data_filter_dtmf": False,
            "df2_mp_data_dropna": False,
            "df2_mp_data_merge": False,
            "df2_mp_data_convert": False,
            "df2_mp_data_drop": False,
            # File Ticks Data Filters
            "df3_mp_data_filter_int": False,
            "df3_mp_data_filter_flt": False,
            "df3_mp_data_filter_obj": False,
            "df3_mp_data_filter_dtmi": False,
            "df3_mp_data_filter_dtmf": False,
            "df3_mp_data_dropna": False,
            "df3_mp_data_merge": True,
            "df3_mp_data_convert": True,
            "df3_mp_data_drop": True,
            # File Rates Data Filters
            "df4_mp_data_filter_int": False,
            "df4_mp_data_filter_flt": False,
            "df4_mp_data_filter_obj": False,
            "df4_mp_data_filter_dtmi": False,
            "df4_mp_data_filter_dtmf": True,
            "df4_mp_data_dropna": False,
            "df4_mp_data_merge": True,
            "df4_mp_data_convert": True,
            "df4_mp_data_drop": True,
        }

        try:
            self.env.override_params({"data": data_overrides})
            logger.info("Data parameters overridden successfully.")
        except Exception as e:
            logger.error("Failed to override data parameters: %s", e)

    def _set_ml_overrides(self):
        """
        Placeholder for overriding ML-related parameters.
        """
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping ML overrides.")
            return
        
        ml_overrides = {
            # Add ML-related overrides here when needed
        }

        if ml_overrides:
            try:
                self.env.override_params({"ml": ml_overrides})
                logger.info("ML parameters overridden successfully.")
            except Exception as e:
                logger.error("Failed to override ML parameters: %s", e)

    def _set_mltune_overrides(self):
        """
        Placeholder for overriding ML tuning parameters.
        """
        if not hasattr(self, "env") or self.env is None:
            logger.error("Environment manager is not initialized. Skipping ML tuning overrides.")
            return
        
        mltune_overrides = {
            # Add ML tuning-related overrides here when needed
        }

        if mltune_overrides:
            try:
                self.env.override_params({"mltune": mltune_overrides})
                logger.info("ML tuning parameters overridden successfully.")
            except Exception as e:
                logger.error("Failed to override ML tuning parameters: %s", e)
