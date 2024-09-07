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
# Import AI packages
#+-------------------------------------------------------------------
#scikit-learn library. It includes utilities for data preprocessing, model evaluation, and model selection.
import scipy as sklearn
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#This line imports the KFold class from scikit-learn, which is often used for cross-validation in machine learning.
from sklearn.model_selection import KFold
#TensorFlow and Keras libraries
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

#+-------------------------------------------------------------------
# Import Local components
#+-------------------------------------------------------------------
import ts as tsmql

