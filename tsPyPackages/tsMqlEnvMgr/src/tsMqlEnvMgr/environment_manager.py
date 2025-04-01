#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: environment_managers.py
File: tsPyPackages/tsMqlGlobalParams/src/tsMqlGlobalParams/environment_managers.py
Description: Set global environment parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""
import logging
logger = logging.getLogger(__name__)
from tsMqlBaseParams.setbase_params import CMqlEnvBaseParams
from tsMqlDataParams import CMqlEnvDataParams
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLTunerParams import CMqlEnvMLTunerParams
from tsMqlAppParams import CMqlEnvAppParams

class CMqlEnvMgr:
    """Central manager for environment parameters."""
    def __init__(self, custom_params=None):
        self.base = CMqlEnvBaseParams()
        self.data = CMqlEnvDataParams()
        self.ml = CMqlEnvMLParams()
        self.mltune = CMqlEnvMLTunerParams()  # Import tuner parameters
        self.app = CMqlEnvAppParams()
        
        self.env_params = {
            "base": self.base.all_params(),
            "data": self.data.all_params(),
            "ml": self.ml.all_params(),
            "mltune": self.mltune.all_params(),
            "app": self.app.all_params()
        }
        
        if custom_params:
            self.override_params(custom_params)

    def get_param(self, category, key, default=None):
        return self.env_params.get(category, {}).get(key, default)

    def get_params(self, category, key, default=None):
        return self.env_params.get(category, {}).get(key, default)

    def override_params(self, new_params):
        """Override parameters with a new dictionary."""
        for category, params in new_params.items():
            logger.info(f"Overriding {category} parameters: {params}")
            if category in self.env_params:
                self.env_params[category].update(params)
                logger.info(f"Overridden {category} parameters: {self.env_params[category]}")

    def split_params(self, *categories):
        """Retrieve only selected parameter categories."""
        return {cat: self.env_params[cat] for cat in categories if cat in self.env_params}

    def list_categories(self):
        """Return a list of available parameter categories."""
        return list(self.env_params.keys())

    def all_params(self):
        """Retrieve all parameters."""
        return self.env_params
