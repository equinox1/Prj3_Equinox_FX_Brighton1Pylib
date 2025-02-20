"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlMLTunerParams.py
File: tsPyPackages/tsMqlMLTunerParams/tsMqlMLTunerParams.py
Description: Load and set machine learning tuner parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""


# Each parameter Class now extends the CEnvCore class and has a 
# DEFAULT_PARAMS dictionary that contains the default values for the parameters.
# The CEnvCore class has a get_params method that returns the parameters
# dictionary and a set_param method that allows you to set a parameter dynamically.
# The CMqlEnvDataParams and CMqlEnvMLParams classes have the DEFAULT_PARAMS dictionary 
# that contains the default values for the parameters. The CMqlEnvDataParams class has
# parameters related to data loading, while the CMqlEnvMLParams class has parameters 
# related to machine learning.

from tsMqlEnvCore import CEnvCore

from datetime import date

def date_helper():
    """Returns the current date and time as a string."""
    return date.today().strftime('%Y-%m-%d %H:%M:%S')


class CMqlEnvMLTunerParams(CEnvCore):
    """Manage machine learning tuner parameters."""
    def __init__(self, **kwargs):
        super().__init__(custom_params=kwargs)  # Ensure proper initialization

         # Set the current date and time  
        self.params["today"] = date_helper()

    DEFAULT_PARAMS = {
        "mp_ml_tunemode": True,
        "tunemodeepochs": 100,
        "seed": 42,
        "tuner_id": 1,
        "train_split": 0.7,
        "validation_split": 0.2,
        "test_split": 0.1,
        "batch_size": 8,
        "input_width": 24,
        "shift": 24,
        "total_window_size": 48,
        "label_width": 1,
        "keras_tuner": "Hyperband",
        "hyperband_iterations": 1,
        "max_epochs": 100,
        "min_epochs": 10,
        "epochs": 2,
        "num_trials": 3,
        "overwrite": True,
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "metrics": ["mean_squared_error"],
        "dropout": 0.2
    }
