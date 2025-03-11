#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: platform_checker.py
File: tsPyPackages/tsMqlPlatform/src/tsMqlPlatform/platform_checker.py
Description: Platform Checker
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: MIT License
"""

import sys
import platform
import os
import logging
from .config import config

# Configure the logger (adjust configuration as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlatformChecker:
    def __init__(self):
        self.env_platform = os.getenv("FORCE_PLATFORM", None)
        self.config_platform = config.get("default_platform", None)

        if self.env_platform:
            logger.info(f"Platform overridden by environment variable: {self.env_platform}")
            self.system = self.env_platform
        elif self.config_platform:
            logger.info(f"Platform overridden by config file: {self.config_platform}")
            self.system = self.config_platform
        else:
            self.system = platform.system()

        logger.info(f"Final detected platform: {self.system}")

    def get_platform(self):
        """Returns the current platform."""
        return self.system

    def is_windows(self):
        """Checks if the platform is Windows."""
        return self.system == "Windows"

    def is_linux(self):
        """Checks if the platform is Linux."""
        return self.system == "Linux"

    def is_macos(self):
        """Checks if the platform is macOS."""
        return self.system == "Darwin"

    def import_platform_dependencies(self):
        """Imports platform-specific dependencies."""
        if self.is_windows():
            try:
                import MetaTrader5 as mt5
                import onnx
                import tf2onnx
                import onnxruntime as ort
                import onnxruntime.backend as backend
                import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
                from onnx import checker
                logger.info("Windows detected. Successfully imported MetaTrader5 and ONNX.")
                return {"mt5": mt5, "onnx": onnx}
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5 or ONNX: {e}")
                sys.exit("[ERROR] Missing dependencies for Windows: MetaTrader5 or ONNX.")
        else:
            logger.info("Non-Windows platform detected. No additional imports required.")
            return None

# Create an instance to use across imports
platform_checker = PlatformChecker()
PLATFORM_DEPENDENCIES = platform_checker.import_platform_dependencies()
