#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: run_platform.py
File: tsPyPackages/tsMqlPlatform/src/tsMqlPlatform/run_platform.py
Description: Run platform checker.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: MIT License
"""
import logging

from tsMqlPlatform import platform_checker, PLATFORM_DEPENDENCIES, config

logger = logging.getLogger(__name__)

logger.info("Starting application...")

class RunPlatform:
    def __init__(self):
        self.debug = config.get("debug", False)
        self.platform_name = platform_checker.get_platform()

        if self.debug:
            print(f"Running on: {self.platform_name}")

        self.mt5 = None  # Initialize mt5 attribute
        self.onnx = None  # Initialize onnx attribute

        if platform_checker.is_windows():
            if PLATFORM_DEPENDENCIES is not None:
                self.mt5 = PLATFORM_DEPENDENCIES.get("mt5")  # Assign to class attributes
                self.onnx = PLATFORM_DEPENDENCIES.get("onnx")

                if self.mt5 and self.onnx:
                    logger.info("MetaTrader5 and ONNX are ready for use.")
                    if self.debug:
                        print("MetaTrader5 and ONNX are available.")
                else:
                    logger.error("MetaTrader5 or ONNX dependencies are missing.")
            else:
                logger.error("PLATFORM_DEPENDENCIES is not initialized.")
        else:
            logger.info("No additional dependencies are needed for this platform.")

    def get_platform(self):
        return self.platform_name

    def get_mt5(self):
        return self.mt5

    def get_onnx(self):
        return self.onnx

    def check_mql_state(self):
        """Return True if both MetaTrader5 and ONNX are available."""
        return self.mt5 is not None and self.onnx is not None

if __name__ == "__main__":
    rp = RunPlatform()
