#!/usr/bin/env python3
# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +------------------------------------------------------------------+

# STEP: Platform settings
# +-------------------------------------------------------------------
# Log timeframe h4 in the data parameters and override the default values
# Log where to src lp-timeframe
# Log override params to file
# GPU and tensor platform

from tsMqlSetup import CMqlSetup
import logging

# Initialize logger
logger = logging.getLogger("Main")
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config

# Initialize platform checker and state
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
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

# Set package options
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

# Equinox environment manager and custom modules
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides
from tsMqlUtilities import CUtilities
from tsMqlReference import CMqlRefConfig
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlMLTune import CMdtuner
from tsMqlMLProcess import CDMLProcess

# Setup logging and tensor platform dependencies
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = obj1_CMqlSetup.get_computation_strategy()

# Format values for data table display
mp_data_tab_rows = 5
mp_data_tab_width = 30
hrows = mp_data_tab_rows
hwidth = mp_data_tab_width

# +-------------------------------------------------------------------
# STEP: End of driving parameters
# +-------------------------------------------------------------------

def main(logger):
    # Ensure that the TensorFlow strategy scope is used
    with strategy.scope():
        # +----------------------------------------------------------
        # STEP: Setup environment
        # +----------------------------------------------------------
        utils_config = CUtilities()
        
        # Set up logging for demonstration purposes
        logging.basicConfig(level=logging.INFO)
         
        # Create an instance of the CMqlOverrides class using the config file
        mql_overrides = CMqlOverrides("config.yaml")
         
        # Retrieve overridden parameters for inspection using the "params" attribute
        data_params = mql_overrides.params.get("data", {})
        feat_params = mql_overrides.params.get("feat", {})
        ml_params = mql_overrides.params.get("ml", {})
        mltune_params = mql_overrides.params.get("mltune", {})
        app_params = mql_overrides.params.get("app", {})
         
        print("Data Parameters:")
        for key, value in data_params.items():
            print(f"  {key}: {value}")
         
        print("\nFeature Parameters:")
        for key, value in feat_params.items():
            print(f"  {key}: {value}")
         
        print("\nML Parameters:")
        for key, value in ml_params.items():
            print(f"  {key}: {value}")
         
        print("\nML Tuning Parameters:")
        for key, value in mltune_params.items():
            print(f"  {key}: {value}")

        print("\nApp Parameters:")
        for key, value in app_params.items():
            print(f"  {key}: {value}")

        # +-------------------------------------------------------------------
        # STEP: Load Reference class and time variables
        # +-------------------------------------------------------------------
        lp_timeframe_name = data_params.get('mp_data_timeframe', 'H4')

        reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
        # Symbol Constants
        PRIMARY_SYMBOL = app_params.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD'))
        SECONDARY_SYMBOL = app_params.get('lp_app_secondary_symbol', app_params.get('mp_app_primary_symbol', 'EURCHF'))
        # Time Constants
        UNIT = reference_config.TIME_CONSTANTS["UNIT"]["SECOND"]
        MINUTE = reference_config.get_timevalue('MINUTE')
        HOUR = reference_config.get_timevalue('HOUR')
        DAY = reference_config.get_timevalue('DAY')
        # Current Time Constants
        CURRENT_TIME = reference_config.get_current_time()
        CURRENTDAY = CURRENT_TIME["CURRENTDAY"]
        CURRENTMONTH = CURRENT_TIME["CURRENTMONTH"]
        CURRENTYEAR = CURRENT_TIME["CURRENTYEAR"]

        # Mql Time Constants
        TIMEZONE = CURRENT_TIME["TIMEZONE"]
        TIMEFRAME = CURRENT_TIME["TIMEFRAME"]
        timeval = HOUR  # hours, used in the window creation
        logger.info(f"Timezone: {TIMEZONE}")
        logger.info(f"Timeframe: {TIMEFRAME}")

        # Fetch individual parameters safely
        rows = data_params.get('mp_data_rows', 1000)
        rowcount = data_params.get('mp_data_rowcount', 10000)
        logger.info(f"Timeframe Name: {lp_timeframe_name}, Rows: {rows}, Rowcount: {rowcount}")

        # +-------------------------------------------------------------------
        # STEP: CBroker Login
        # +-------------------------------------------------------------------
        logger.info("PARAM HEADER: MP_APP_BROKER:", app_params.get('mp_app_broker'))
        broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
        mqqlobj = broker_config.run_mql_login()
        if mqqlobj is True:
            logger.info("Successfully logged in to MetaTrader 5.")
        else:
            logger.info(f"Failed to login. Error code: {mqqlobj}")

        # +-------------------------------------------------------------------
        # STEP: Initiate the data loader and data process classes
        # +-------------------------------------------------------------------
        # Retrieve broker file paths
        data_loader_config = CDataLoader()
        data_process_config = CDataProcess(mp_unit=UNIT)
        ml_process_config = CDMLProcess()

        # +-------------------------------------------------------------------
        # STEP: Data loading and processing
        # +-------------------------------------------------------------------
        mp_data_history_size = data_params.get('mp_data_history_size', 1)  # Default to 1 if not set
        # Set the UTC time for the data
        mv_data_utc_from = data_loader_config.set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        mv_data_utc_to = data_loader_config.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        logger.info(f"Main: UTC From: {mv_data_utc_from}")
        logger.info(f"Main: UTC To: {mv_data_utc_to}")

        # Load the data
        data_loader_config = CDataLoader(
            lp_utc_from=mv_data_utc_from,
            lp_utc_to=mv_data_utc_to,
            lp_timeframe=lp_timeframe_name,
            lp_app_primary_symbol=PRIMARY_SYMBOL,
            lp_app_rows=rows,
            lp_app_rowcount=rowcount
        )
        df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = data_loader_config.run_dataloader_services()
        logger.info(f"Loaded: Data API Ticks: {df_api_ticks.shape}, Data API Rates: {df_api_rates.shape}, "
                    f"Data File Ticks: {df_file_ticks.shape}, Data File Rates: {df_file_rates.shape}")

        # +-------------------------------------------------------------------
        # STEP: Run data process manipulation
        # +-------------------------------------------------------------------
        df_api_ticks = data_process_config.run_dataprocess_services(df=df_api_ticks, df_name='df_api_ticks')
        df_api_rates = data_process_config.run_dataprocess_services(df=df_api_rates, df_name='df_api_rates')
        df_file_ticks = data_process_config.run_dataprocess_services(df=df_file_ticks, df_name='df_file_ticks')
        df_file_rates = data_process_config.run_dataprocess_services(df=df_file_rates, df_name='df_file_rates')

        # Log processed data summaries
        utils_config.run_mql_print(df=df_api_ticks, df_name='df_api_ticks', hrows=hrows, colwidth=hwidth, app='data processing')
        utils_config.run_mql_print(df=df_api_rates, df_name='df_api_rates', hrows=hrows, colwidth=hwidth, app='data processing')
        utils_config.run_mql_print(df=df_file_ticks, df_name='df_file_ticks', hrows=hrows, colwidth=hwidth, app='data processing')
        utils_config.run_mql_print(df=df_file_rates, df_name='df_file_rates', hrows=hrows, colwidth=hwidth, app='data processing')

        datafile = df_file_rates
        # Further processing or model training could occur here.

        
        """
        # +-------------------------------------------------------------------
        # STEP: add The time index to the data
        # +-------------------------------------------------------------------
        # Limit the columns to just the time column and all features
        column_features = datafile.columns[1:]
        logger.info(f"Column Features: {column_features}")
        datafile = datafile[[datafile.columns[0]] + list(datafile.columns[1:])]
        logger.info(f"Data File Shape: {datafile.head(hrows)}")

        # +-------------------------------------------------------------------
        # STEP: Create Window Parameters
        # +-------------------------------------------------------------------
        # 1: 24 HOURS/24 HOURS prediction window
        logger.info("Creating the 24 hour prediction timeval{timeval},hour {HOUR}")
        past_width, future_width, pred_width = ml_process_config.create_ml_window(timeval=HOUR)
        logger.info(f"Past Width: {past_width}, Future Width: {future_width}, Prediction Width: {pred_width}")
        
        # +-------------------------------------------------------------------
        # STEP: Generate X and y from the Time Series
        # +-------------------------------------------------------------------
        feature1=ml_params.get('Feature1', ml_params.get('Feature1', 'Feature1'))
        feature1_scaled=ml_params.get('Feature1_scaled', ml_params.get('Feature1_scaled', 'Feature1_scaled'))
        label1=ml_params.get('Label1', ml_params.get('Label1', 'Label1'))
        logger.info(f"Main: Feature1: {feature1}, Feature1 Scaled: {feature1_scaled}, Label1: {label1},window size: {past_width} , using past width")

        datafile_X,datafile_y = ml_process_config.create_XY_unscaled_feature_sequence(datafile, target_col=feature1, window_size=past_width)
        logger.info(f"Datafile X: {datafile_X.shape}, Datafile y: {datafile_y.shape}")
     
        # +-------------------------------------------------------------------
        # STEP: Split the data into training and test sets Fixed Partitioning
        # +-------------------------------------------------------------------
        seed= mltune_params.get('seed', 42)
        # Split the dataset
        X_train, X_val, X_test, y_train, y_val, y_test = ml_process_config.split_dataset(datafile_X,datafile_y , random_state=seed)
        
        # +-------------------------------------------------------------------
        # STEP:convert numpy dataset to TF dataset
        # +-------------------------------------------------------------------
        # initiate the object using a window generatorwindow is not  used in this model Parameters
        tf_batch_size = ml_params.get('tf_batch_size', 32)
        # Preprocess to avoid datetime columns
        logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
      
    
        # Convert to TensorFlow datasets
        train_dataset, val_dataset, test_dataset = ml_process_config.convert_to_tfds(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=tf_batch_size)
        logger.info(f"Train Dataset: {train_dataset}, Val Dataset: {val_dataset}, Test Dataset: {test_dataset}")
       
        # +-------------------------------------------------------------------
        # STEP: Shapes: add tensor values for model input
        # +-------------------------------------------------------------------
        train_shape, val_shape, test_shape = get_dataset_shapes(train_dataset, val_dataset, test_dataset)
        input_shape = X_train.shape[1:]  # Shape of a single sample (time_steps, features)
        label_shape = y_train.shape[1:]  # Shape of a single label (future_steps)
         
      
         # +-------------------------------------------------------------------
         # STEP: Tune best model Hyperparameter tuning and model setup
         # +-------------------------------------------------------------------
         # Hyperparameter configuration
         obj1_TunerParams = CMdtunerHyperModel(
            tuner_params=tunerparams["tunerparams"],
            input_shape=input_shape,
            data_input_shape=input_shape,
            output_shape=output_label_shape,
            input_batch_size=batch_size,
            tunemode=True,
            tunemodeepochs=True,
            model_summary=True,
            batch_size=batch_size,
            epochs=15
         )  
         hypermodel_params = obj1_TunerParams.get_hypermodel_params()
         logger.info("Tuner Parameters:", hypermodel_params)  # Print the tuner parameters

         
         #instansiate tuner claa
         obj1_CMdtuner = CMdtuner(
                     hypermodel_params=hypermodel_params,
                     traindataset=train_dataset,
                     valdataset=val_dataset,
                     testdataset=test_dataset,
                     castmode='float32',
                     
                     )
         #initialize tuner
         obj1_CMdtuner.initialize_tuner()
         
         # Check and load the model
         best_model = obj1_CMdtuner.check_and_load_model(mp_ml_mbase_path, ftype='tf')

         # If no model or a hard run is required, run the search
         runtuner = False

         if best_model is None:
               logger.info("Running the tuner search bo model")
               runtuner = mt.run_search()
               mt.export_best_model(ftype='tf')
         elif mp_ml_hard_run:  
               logger.info("Running the tuner search hard run")
               runtuner = mt.run_search()
               mt.export_best_model(ftype='tf')
         else:
               logger.info("Best model loaded successfully")
               runtuner = True

         # +-------------------------------------------------------------------
         # STEP: Train and evaluate the best model
         # +-------------------------------------------------------------------
         logger.info("Model: Loading file from directory", mp_ml_mbase_path,"Model: filename", mp_ml_project_name)   
         if (load_model := mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')) is not None:
               best_model = load_model
               logger.info("Model: Best model: ", best_model.name)
               logger.info("best_model.summary()", best_model.summary())
               
               # Fit the label scaler on the training labels
               
               # Model Training
               tf.keras.backend.clear_session(free_memory=True)

               logger.info("Training the best model...")
               best_model.fit(
                  train_dataset,
                  validation_data=val_dataset,
                  batch_size=batch_size,
                  epochs=mp_ml_tf_param_epochs
               )
               logger.info("Training completed.")
               
               # Model Evaluation
               logger.info("Evaluating the model...")
               val_metrics = best_model.evaluate(val_dataset, verbose=1)
               test_metrics = best_model.evaluate(test_dataset, verbose=1)

               logger.info(f"Validation Metrics - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")
               logger.info(f"Test Metrics - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}")

               # Fit the label scaler on the training labels
               label_scaler.fit(y_train.reshape(-1, 1))

               # Predictions and Scaling
               logger.info("Running predictions and scaling...")
               predicted_fx_price = best_model.predict(test_dataset)
               predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)

               real_fx_price = label_scaler.inverse_transform(y_test.reshape(-1, 1))
               logger.info("Predictions and scaling completed.")
               # +-------------------------------------------------------------------
               # STEP: Performance Check
               # +-------------------------------------------------------------------
               # Evaluation and visualization
               # Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
               # The lower the MSE, the better the model.

               # Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values. 
               # Like MSE, lower values indicate better model performance.

               # R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
               # dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a 
               # perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the label
               # variable. Negative values indicate poor model performance.
               # Check for NaN values and handle them
               if np.isnan(real_fx_price).any() or np.isnan(predicted_fx_price).any():
                  logger.info("Warning: NaN values found in input data. Handling NaNs by removing corresponding entries.")
                  mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)
                  real_fx_price = real_fx_price[mask]
                  predicted_fx_price = predicted_fx_price[mask]
               
               mse = mean_squared_error(real_fx_price, predicted_fx_price)
               mae = mean_absolute_error(real_fx_price, predicted_fx_price)
               r2 = r2_score(real_fx_price, predicted_fx_price)
               logger.info(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
               logger.info(f"Mean Squared Error: The lower the MSE, the better the model: {mse}")
               logger.info(f"Mean Absolute Error: The lower the MAE, the better the model: {mae}")
               logger.info(f"R2 Score: The closer to 1, the better the model: {r2}")

               plt.plot(real_fx_price, color='red', label='Real FX Price')
               plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
               plt.title('FX Price Prediction')
               plt.xlabel('Time')
               plt.ylabel('FX Price')
               plt.legend()
               plt.savefig(mp_ml_base_path + '/' + 'plot.png')
               if mp_ml_show_plot:
                  plt.show()
               logger.info("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')

               if ONNX_save:
                  # Save the model to ONNX
                  
                  # Define the output path
                  mp_output_path = mp_ml_data_path + f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx"
                  logger.info(f"Output Path: {mp_output_path}")

                  # Convert Keras model to ONNX
                  opset_version = 17  # Choose an appropriate ONNX opset version

                  # Assuming your model has a single input
                  spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]
                  onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)

                  # Save the ONNX model
                  onnx.save_model(onnx_model, mp_output_path)
                  logger.info(f"Model saved to {mp_output_path}")

                  # Verify the ONNX model
                  checker.check_model(onnx_model)
                  logger.info("ONNX model is valid.")

                  # Check ONNX Runtime version
                  logger.info("ONNX Runtime version:", ort.__version__)

                  # finish
                  mt5.shutdown()
                  logger.info("Finished")
         else:
               logger.info("No data loaded")
         mt5.shutdown()
         logger.info("Finished")    
"""
if __name__ == "__main__":
    main(logger)