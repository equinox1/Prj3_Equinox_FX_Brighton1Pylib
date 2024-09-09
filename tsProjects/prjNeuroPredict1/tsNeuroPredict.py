#+------------------------------------------------------------------+
#|                                                 Neuropredict2.py |
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Import Standard Python packages
#+-------------------------------------------------------------------
import sys
from datetime import datetime
import pytz

import pandas as pd
#NumPy is widely used for numerical operations in Python.
import numpy as np
#+-------------------------------------------------------------------
# Import MQL packages
#+-------------------------------------------------------------------
import MetaTrader5 as mt5

#+-------------------------------------------------------------------
# Import AI packages SciKit Learns
#+-------------------------------------------------------------------
#sklearn library. It includes utilities for data preprocessing, model evaluation, and model selection.
import sklearn 
from sklearn  import datasets, svm, metrics
from sklearn .model_selection import train_test_split, cross_val_score
from sklearn .preprocessing import StandardScaler
from sklearn .metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from sklearn, which is often used for cross-validation in machine learning.
from sklearn .model_selection import KFold

#======================================================
# Import AI packages TensorFlow and Keras libraries
#======================================================
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2

#+-------------------------------------------------------------------
# Import Local components
#+-------------------------------------------------------------------
import tsMqlData as tsd1
import tsMqlML as tsd2
