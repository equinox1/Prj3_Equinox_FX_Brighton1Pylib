"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: setglobal_params.py
File: tsPyPackages/tsMqlGlobalParams/src/tsMqlGlobalParams/setglobal_params.py
Description: Set global environment parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""
from tsMqlBaseParams import CMqlEnvBaseParams
from tsMqlDataParams import CMqlEnvDataParams
from tsMqlMLParams import CMqlEnvMLParams 
from tsMqlMLTunerParams import CMqlEnvMLTunerParams

class CMqlEnvGlobal:
    """Manage global environment parameters."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.base_params = CMqlEnvBaseParams()
        self.data_params = CMqlEnvDataParams()
        self.ml_params = CMqlEnvMLParams()
        self.tuner_params = CMqlEnvMLTunerParams()
    
    def get_params(self):
        """Returns all parameters in a structured dictionary."""
        return {
            'genparams': self.params,
            'dataparams': self.data_params.get_params(),
            'mlearnparams': self.ml_params.get_params(),
            'tunerparams': self.tuner_params.get_params(),
        }
