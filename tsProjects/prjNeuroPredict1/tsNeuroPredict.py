#+------------------------------------------------------------------+
#|                                                 neuropredict2.py |
#|                                                    tony shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# import standard python packages
#+-------------------------------------------------------------------
import sys
from datetime import datetime
import pytz

import pandas as pd
#numpy is widely used for numerical operations in python.
import numpy as np
#+-------------------------------------------------------------------
# import mql packages
#+-------------------------------------------------------------------
import MetaTrader5 as mt5

#+-------------------------------------------------------------------
# import ai packages scikit learns
#+-------------------------------------------------------------------
#sklearn library. it includes utilities for data preprocessing, model evaluation, and model selection.
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from scikit-learn, which is often used for cross-validation in machine learning.
from sklearn.model_selection import KFold

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2

#+-------------------------------------------------------------------
# import local components
#+-------------------------------------------------------------------
#import tsmqldata as tsd1

