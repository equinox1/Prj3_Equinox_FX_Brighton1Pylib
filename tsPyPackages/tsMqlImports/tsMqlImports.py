# gpu and tensor platform
from tsMqlSetup import CMqlSetup
import logging

# Initialize logger
logger = logging.getLogger("Main")
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
# System packages
import os
import pathlib
from pathlib import Path
import json
from datetime import datetime, date
import pytz

# Data packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Machine Learning packages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# set package options
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr
#Reference class
from tsMqlReference import CMqlRefConfig
# Equinox sub packages
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
# Equinox ML packages
from tsMqlMLTune import CMdtuner
from tsMqlMLParams import CMqlEnvMLParams