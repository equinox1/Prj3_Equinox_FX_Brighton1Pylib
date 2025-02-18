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

"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlDataParams.py
File: tsPyPackages/tsMqlDataParams/tsMqlDataParams.py
Description: Load adat files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""
# Each parameter Class now extends the BaseParamManager class and has a 
# DEFAULT_PARAMS dictionary that contains the default values for the parameters.
# The BaseParamManager class has a get_params method that returns the parameters
# dictionary and a set_param method that allows you to set a parameter dynamically.
# The CMqlEnvDataParams and CMqlEnvMLParams classes have the DEFAULT_PARAMS dictionary 
# that contains the default values for the parameters. The CMqlEnvDataParams class has
# parameters related to data loading, while the CMqlEnvMLParams class has parameters 
# related to machine learning.

class CMqlEnvMLTunerParams(BaseParamManager):
    DEFAULT_PARAMS = {
        "mp_ml_tunemode": True,
        "tunemodeepochs": 100,
        "modelsummary": False,
        "today": date.today().strftime('%Y-%m-%d %H:%M:%S'),
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