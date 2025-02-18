"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename : tsMqlBaseParamMgmt.py
File : tsPyPackages/tsMqlBaseParamMgmt/tsMqlBaseParamMgmt.py
Description :Base manager for all params
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

import logging
from datetime import date

logger = logging.getLogger("tsMqlParams")
logging.basicConfig(level=logging.INFO)

class ParamMgr:
    """Base class for parameter management."""
    
    DEFAULT_PARAMS = {}

    def __init__(self, **overrides):
        """Initialize parameters, allowing overrides."""
        self.params = {key: overrides.get(key, default) for key, default in self.DEFAULT_PARAMS.items()}
        
    def get_params(self):
        """Return a dictionary of all parameters."""
        return self.params

    def set_param(self, key, value):
        """Set a parameter dynamically."""
        if key in self.params:
            self.params[key] = value
        else:
            logger.warning(f"Unknown parameter '{key}' - Consider adding it to defaults.")

