#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlEnvCore.py
File: tsPyPackages/tsMqlEnvCore/tsMqlEnvCore.py
Description: Load and add files and data parameters. Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
"""

import logging
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class CEnvCore:
    """Common functionality shared across parameter classes."""
    
    DEFAULT_PARAMS = {}  # Ensure this is a class attribute
    
    def __init__(self, custom_params=None):
        self.params = self.DEFAULT_PARAMS.copy()  # Initialize instance params from the class attribute
        if custom_params:
            self.params.update(custom_params)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def get_params(self, key, default=None):
        return self.params.get(key, default)

    def set(self, key, value):
        self.params[key] = value

    def all_params(self):
        return self.params  # Returns the dictionary of all parameters
