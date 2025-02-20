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
import logging
logger = logging.getLogger("setglobal_params")
logging.basicConfig(level=logging.INFO)

from tsMqlBaseParams.setbase_params import CMqlEnvBaseParams
from tsMqlDataParams import CMqlEnvDataParams
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLTunerParams import CMqlEnvMLTunerParams

class CMqlGlobalParams:
    """Manage global environment parameters."""
    def __init__(self, **kwargs):
        self.params = kwargs
        self.base_params = CMqlEnvBaseParams(**kwargs)
        self.data_params =  CMqlEnvDataParams()
        self.ml_params =  CMqlEnvMLParams()
        self.tuner_params =  CMqlEnvMLTunerParams()
    
    def get_params(self):
        """Returns all parameters in a structured dictionary."""
        base = self.base_params.get_params()
        logger.info(f"Base Params Retrieved: {base}")
        return {
            'baseparams': base,
            'dataparams': self.data_params,
            'mlearnparams': self.ml_params,
            'tunerparams': self.tuner_params,
        }
